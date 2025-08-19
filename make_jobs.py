#!/usr/bin/env python3
import argparse, itertools, os, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, required=True, help="Comma-separated datasets (e.g., cora,citeseer)")
    p.add_argument("--model", type=str, default="tfe", help="Model tag (kept for compatibility; not used)")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--K", type=int, default=5, help="Polynomial order hint; used to fill --hops when not provided")
    p.add_argument("--bands", type=int, default=2)
    p.add_argument("--tau", type=str, default="", help="Unused for TFE; kept for compatibility")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3, help="Linear LR")
    p.add_argument("--wd", type=float, default=5e-5, help="Linear WD")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    # Build per-band hops (repeat K for all bands by default)
    hops = ",".join([str(args.K)] * args.bands)

    # Prepare commands (one per row) — we call tfe_training.py directly
    rows = [("id", "cmd")]
    jid = 0
    for ds in datasets:
        for seed in range(args.seeds):
            cmd = (
                "python3 tfe_training.py "
                f"--dataset {ds} "
                f"--hidden {args.hidden} "
                f"--layers {args.layers} "
                f"--pro_dropout {args.dropout} --lin_dropout {args.dropout} "
                f"--lr_lin {args.lr} --wd_lin {args.wd} "
                f"--epochs {args.epochs} --patience {args.patience} "
                f"--gf sym --combine sum "
                f"--bands {args.bands} --hops {hops} "
                f"--runs 1 --optimizer Adam --device 0 "
                f"--hop_lp {args.K} --hop_hp {args.K} "
                f"--seed {seed} "
            )
            rows.append((str(jid), cmd))
            jid += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rid, cmd in rows:
            f.write(f"{rid}\t{cmd}\n")

if __name__ == "__main__":
    main()
