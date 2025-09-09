#!/usr/bin/env python3
# tools/eval_orchestrator.py
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# === HARD LIMITS (per your spec) ===
MAX_CONCURRENT_SBATCH = 10   # do not change unless you change the spec
MAX_CMDS_PER_JOB      = 20   # ≤20 execs per worker job

# === Defaults you asked for (can be overridden via CLI flags if you want) ===
DEFAULT_BANDS_SET      = [2, 3, 4, 5]
DEFAULT_BANDWIDTH_SET  = [1, 3, 5, 10]
DEFAULT_DATASET        = "squirrel"
DEFAULT_EPOCHS         = 1000
DEFAULT_PATIENCE       = 200
DEFAULT_HIDDEN         = 512
DEFAULT_LAYERS         = 2
DEFAULT_DEVICE         = 0
DEFAULT_RUNS           = 1
DEFAULT_OPTIM          = "Adam"
DEFAULT_HOP_LP         = 2
DEFAULT_HOP_HP         = 5
DEFAULT_PRO_DROPOUT    = 0.6
DEFAULT_LIN_DROPOUT    = 0.0
DEFAULT_ETA            = -0.5
DEFAULT_Gf             = "sym"
DEFAULT_ACTIVATION     = True
DEFAULT_FULL           = True
DEFAULT_RANDOM_SPLIT   = True
DEFAULT_COMBINE        = "sum"
DEFAULT_SEED           = 42
DEFAULT_HOP_VALUE      = 5  # base K used to fill --hops as "5,5,...,5"

# Paths
SLURM_WORKER = "slurm/worker.sbatch"     # next file we'll add
SLURM_ARRAY  = "slurm/array.sh"          # thin wrapper; added in step 3
BATCH_DIR    = Path("batches")
RESULTS_DIR  = Path("results")
RECORDS_DIR  = RESULTS_DIR / "records"

def pairs_from_bandwidths(options: Sequence[int]) -> List[Tuple[int,int]]:
    """All ordered (lo,hi) pairs."""
    return [(lo, hi) for lo in options for hi in options]

def combos_for_bands(bands: int, bandwidth_options: Sequence[int]) -> List[Tuple[Tuple[int,int], ...]]:
    """
    For given bands, build all multisets (with replacement) of size midbands = bands - 2,
    drawn from the (lo,hi) pairs. Order inside the multiset doesn't matter.
    """
    midbands = bands - 2
    if midbands <= 0:
        return [tuple()]  # only LP & HP
    all_pairs = pairs_from_bandwidths(bandwidth_options)
    return list(itertools.combinations_with_replacement(all_pairs, midbands))

def flatten_bandwidth_combo(combo: Tuple[Tuple[int,int], ...]) -> str:
    """( (l1,h1), (l2,h2), ... ) -> 'l1,h1,l2,h2,...' """
    flat: List[str] = []
    for lo, hi in combo:
        flat.extend([str(lo), str(hi)])
    return ",".join(flat)

def hops_csv(bands: int, hop_value: int = DEFAULT_HOP_VALUE) -> str:
    return ",".join([str(hop_value)] * bands)

def make_leaf_cmd(
    dataset: str, bands: int, bw_combo_csv: str, hops_csv_str: str,
    epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE, hidden=DEFAULT_HIDDEN, layers=DEFAULT_LAYERS,
    device=DEFAULT_DEVICE, runs=DEFAULT_RUNS, optimizer=DEFAULT_OPTIM, hop_lp=DEFAULT_HOP_LP,
    hop_hp=DEFAULT_HOP_HP, pro_dropout=DEFAULT_PRO_DROPOUT, lin_dropout=DEFAULT_LIN_DROPOUT,
    eta=DEFAULT_ETA, gf=DEFAULT_Gf, activation=DEFAULT_ACTIVATION, full=DEFAULT_FULL,
    random_split=DEFAULT_RANDOM_SPLIT, combine=DEFAULT_COMBINE, seed=DEFAULT_SEED
) -> str:
    """
    Build the exact sbatch line you gave, filling --bands/--hops/--bandwidths per combo.
    We submit a *leaf* job via slurm/array.sh (thin wrapper around your run).
    """
    bool_str = lambda b: "True" if b else "False"
    line = (
        f"sbatch {SLURM_ARRAY} "
        f"--dataset {dataset} --epochs {epochs} --patience {patience} "
        f"--hidden {hidden} --layers {layers} --device {device} --runs {runs} "
        f"--optimizer {optimizer} --hop_lp {hop_lp} --hop_hp {hop_hp} "
        f"--pro_dropout {pro_dropout} --lin_dropout {lin_dropout} "
        f"--eta {eta} --bands {bands} --lr_adaptive 0.1 --wd_adaptive 0.05 "
        f"--lr_adaptive2 0.0 --wd_adaptive2 0.0 "
        f"--lr_lin 0.005 --wd_lin 0.0 --gf {gf} "
        f"--activation {bool_str(activation)} --full {bool_str(full)} "
        f"--random_split {bool_str(random_split)} --combine {combine} "
        f"--seed {seed} --hops {hops_csv_str}"
    )
    if bw_combo_csv:
        line += f" --bandwidths {bw_combo_csv}"
    return line

def chunk(lst: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), size):
        yield list(lst[i:i+size])

def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

def submit_worker(batch_file: Path) -> int:
    """sbatch slurm/worker.sbatch batches/batch_XXXX.cmds -> returns jobid"""
    cp = run_cmd(["sbatch", SLURM_WORKER, str(batch_file)])
    # stdout typically: "Submitted batch job 123456"
    m = re.search(r"Submitted batch job\s+(\d+)", cp.stdout.strip())
    if not m:
        raise RuntimeError(f"Could not parse sbatch output:\n{cp.stdout}\n{cp.stderr}")
    return int(m.group(1))

def active_worker_count(user: str) -> int:
    # Match by job-name set inside worker.sbatch ("eval_worker")
    try:
        cp = run_cmd(["squeue", "-h", "-u", user, "-n", "eval_worker", "-o", "%i"])
        # one job id per line
        ids = [ln for ln in cp.stdout.splitlines() if ln.strip()]
        return len(ids)
    except subprocess.CalledProcessError:
        # If squeue not available or user has no jobs, treat as 0
        return 0

def wait_for_some_capacity(user: str, target_free_slots: int = 1, poll_sec: int = 15):
    """Block until at least 'target_free_slots' workers can be submitted (<=MAX_CONCURRENT_SBATCH in queue)."""
    while True:
        act = active_worker_count(user)
        free = max(0, MAX_CONCURRENT_SBATCH - act)
        if free >= target_free_slots:
            return
        time.sleep(poll_sec)

def aggregate_results(records_dir: Path, out_dir: Path):
    """
    Expect JSONL files containing one object per execution with keys:
      dataset, bands, bandwidths, acc_mean, acc_std, seed, runs, job_id, ...
    We produce:
      - results/aggregate.csv  (all rows)
      - results/band_<b>.csv   (per-band)
      - print "best per band" summary
    """
    rows = []
    for p in records_dir.glob("*.jsonl"):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # normalize
                rows.append({
                    "dataset":    rec.get("dataset"),
                    "bands":      int(rec.get("bands")),
                    "bandwidths": rec.get("bandwidths"),  # e.g. "l1,h1,l2,h2"
                    "hops":       rec.get("hops"),
                    "seed":       rec.get("seed"),
                    "runs":       rec.get("runs"),
                    "acc_mean":   float(rec.get("acc_mean")),
                    "acc_std":    float(rec.get("acc_std", 0.0)),
                    "job_id":     rec.get("job_id"),
                })

    if not rows:
        print("[aggregate] No JSONL records found; did you add result writing in tfe_training.py?")
        return

    # Write aggregate.csv
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_csv = out_dir / "aggregate.csv"
    headers = ["dataset","bands","bandwidths","hops","seed","runs","acc_mean","acc_std","job_id"]
    with agg_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join([str(r[h]) for h in headers]) + "\n")

    # Per-band files + best per band
    by_band = {}
    for r in rows:
        by_band.setdefault(r["bands"], []).append(r)

    best_lines = []
    for b, lst in sorted(by_band.items()):
        lst_sorted = sorted(lst, key=lambda x: (-x["acc_mean"], x["acc_std"]))
        # per-band CSV
        pb_csv = out_dir / f"band_{b}.csv"
        with pb_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for r in lst_sorted:
                f.write(",".join([str(r[h]) for h in headers]) + "\n")
        # best line
        best = lst_sorted[0]
        best_lines.append(f"bands={b:>2}  acc_mean={best['acc_mean']:.4f}  acc_std={best['acc_std']:.4f}  bw={best['bandwidths']}")

    print("\n=== Best per band ===")
    for ln in best_lines:
        print(ln)
    print(f"\nWrote: {agg_csv} and results/band_<b>.csv")

def main():
    ap = argparse.ArgumentParser(description="Grid-eval orchestrator: batch to ≤20 cmds, keep 10 workers in flight.")
    ap.add_argument("--bands-set", type=str, default=",".join(map(str, DEFAULT_BANDS_SET)),
                    help="Comma-separated bands to explore, e.g. 2,3,4,5")
    ap.add_argument("--bandwidth-set", type=str, default=",".join(map(str, DEFAULT_BANDWIDTH_SET)),
                    help="Bandwidth options, e.g. 1,3,5,10")
    ap.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--device", type=int, default=DEFAULT_DEVICE)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--optimizer", type=str, default=DEFAULT_OPTIM)
    ap.add_argument("--hop-lp", type=int, default=DEFAULT_HOP_LP)
    ap.add_argument("--hop-hp", type=int, default=DEFAULT_HOP_HP)
    ap.add_argument("--pro-dropout", type=float, default=DEFAULT_PRO_DROPOUT)
    ap.add_argument("--lin-dropout", type=float, default=DEFAULT_LIN_DROPOUT)
    ap.add_argument("--eta", type=float, default=DEFAULT_ETA)
    ap.add_argument("--gf", type=str, default=DEFAULT_Gf)
    ap.add_argument("--activation", type=str, default=str(DEFAULT_ACTIVATION))
    ap.add_argument("--full", type=str, default=str(DEFAULT_FULL))
    ap.add_argument("--random-split", type=str, default=str(DEFAULT_RANDOM_SPLIT))
    ap.add_argument("--combine", type=str, default=DEFAULT_COMBINE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--hop-value", type=int, default=DEFAULT_HOP_VALUE, help="K used to fill --hops per band")
    ap.add_argument("--dry-run", action="store_true", help="Only create batches; do not submit.")
    ap.add_argument("--skip-aggregate", action="store_true", help="Do not aggregate at the end.")
    args = ap.parse_args()

    # Parse sets
    bands_set     = sorted({int(x) for x in args.bands_set.split(",") if x.strip()})
    bandwidth_set = sorted({int(x) for x in args.bandwidth_set.split(",") if x.strip()})

    # Build all leaf sbatch commands
    commands: List[str] = []
    for b in bands_set:
        combos = combos_for_bands(b, bandwidth_set)
        for combo in combos:
            bw_csv   = flatten_bandwidth_combo(combo)
            hops_csv_str = hops_csv(b, args.hop_value)
            cmd = make_leaf_cmd(
                dataset=args.dataset, bands=b, bw_combo_csv=bw_csv, hops_csv_str=hops_csv_str,
                epochs=args.epochs, patience=args.patience, hidden=args.hidden, layers=args.layers,
                device=args.device, runs=args.runs, optimizer=args.optimizer, hop_lp=args.hop_lp,
                hop_hp=args.hop_hp, pro_dropout=args.pro_dropout, lin_dropout=args.lin_dropout,
                eta=args.eta, gf=args.gf, activation=(args.activation.lower()=="true"),
                full=(args.full.lower()=="true"), random_split=(args.random_split.lower()=="true"),
                combine=args.combine, seed=args.seed
            )
            commands.append(cmd)

    if not commands:
        print("No commands generated — check your sets.")
        return

    # Write batches (≤20 per file)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_files: List[Path] = []
    for idx, group in enumerate(chunk(commands, MAX_CMDS_PER_JOB), start=1):
        p = BATCH_DIR / f"batch_{idx:04d}.cmds"
        with p.open("w", encoding="utf-8") as f:
            for line in group:
                f.write(line + "\n")
        batch_files.append(p)

    print(f"[plan] Generated {len(commands)} leaf commands across {len(batch_files)} batch files in {BATCH_DIR}/")

    if args.dry_run:
        print("[dry-run] Skipping submission.")
        return

    # Submit up to 10 workers at a time, refilling as they finish
    user = os.environ.get("USER", "")
    pending = list(batch_files)
    launched: List[Tuple[int, Path]] = []

    # Initial fill
    while pending and len(launched) < MAX_CONCURRENT_SBATCH:
        bf = pending.pop(0)
        jid = submit_worker(bf)
        launched.append((jid, bf))
        print(f"[submit] {bf.name} -> job {jid}")

    # Refill loop
    while pending:
        wait_for_some_capacity(user, target_free_slots=1, poll_sec=20)
        # There is capacity for at least one more
        bf = pending.pop(0)
        jid = submit_worker(bf)
        launched.append((jid, bf))
        print(f"[submit] {bf.name} -> job {jid}")

    # Wait until all workers finish
    while active_worker_count(user) > 0:
        print("[wait] Workers still running... polling in 30s")
        time.sleep(30)

    print("[done] All worker jobs finished.")

    # Aggregate
    if not args.skip_aggregate:
        aggregate_results(RECORDS_DIR, RESULTS_DIR)

if __name__ == "__main__":
    main()
