#!/usr/bin/env python3
import csv, os
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
AGG = RESULTS_DIR / "aggregate.csv"
PLOTS = RESULTS_DIR / "plots"
MD = RESULTS_DIR / "summary.md"

def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            # normalize types
            r["bands"] = int(r["bands"])
            r["acc_mean"] = float(r["acc_mean"])
            r["acc_std"] = float(r.get("acc_std", 0.0))
            rows.append(r)
    return rows

def best_per_band(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["bands"]].append(r)
    best = {}
    for b, lst in by.items():
        lst_sorted = sorted(lst, key=lambda x: (-x["acc_mean"], x["acc_std"]))
        best[b] = lst_sorted[0]
    return best, by

def plot_best_per_band(best_map):
    PLOTS.mkdir(parents=True, exist_ok=True)
    bands = sorted(best_map.keys())
    vals = [best_map[b]["acc_mean"]*100.0 for b in bands]  # percent
    plt.figure(figsize=(6, 4))
    plt.bar([str(b) for b in bands], vals)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Bands")
    plt.title("Best accuracy per band")
    out = PLOTS / "best_per_band.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out

def plot_topN_per_band(by_band, N=20):
    outs = []
    for b in sorted(by_band.keys()):
        lst_sorted = sorted(by_band[b], key=lambda x: (-x["acc_mean"], x["acc_std"]))[:N]
        labels = [r["bandwidths"] or "(none)"]  # ensure not empty
        labels = [ (r["bandwidths"] if r["bandwidths"] else "(none)") for r in lst_sorted ]
        vals = [r["acc_mean"]*100.0 for r in lst_sorted]
        plt.figure(figsize=(10, max(3, 0.35*len(vals))))
        plt.barh(range(len(vals)), vals)
        plt.yticks(range(len(vals)), labels)
        plt.xlabel("Accuracy (%)")
        plt.title(f"Top {len(vals)} combos — bands={b}")
        plt.gca().invert_yaxis()
        out = PLOTS / f"band_{b}_top{len(vals)}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        outs.append(out)
    return outs

def write_markdown(best_map, plot_main, per_band_plots):
    with open(MD, "w", encoding="utf-8") as f:
        f.write("# Evaluation Summary\n\n")
        f.write("## Best per band\n\n")
        f.write("| Bands | Accuracy (%) | Bandwidths |\n|---:|---:|:---|\n")
        for b in sorted(best_map.keys()):
            br = best_map[b]
            f.write(f"| {b} | {br['acc_mean']*100:.2f} ± {br['acc_std']*100:.2f} | {br['bandwidths'] or '(none)'} |\n")
        f.write("\n")
        f.write(f"![Best per band]({plot_main.as_posix()})\n\n")
        f.write("## Per-band top combinations\n\n")
        for p in per_band_plots:
            f.write(f"![{p.stem}]({p.as_posix()})\n\n")

def main():
    if not AGG.exists():
        print(f"[plot] {AGG} not found. Run the orchestrator (which writes aggregate.csv) first.")
        return
    rows = read_csv(AGG)
    best_map, by_band = best_per_band(rows)
    plot_main = plot_best_per_band(best_map)
    band_plots = plot_topN_per_band(by_band, N=20)
    write_markdown(best_map, plot_main, band_plots)
    print(f"[plot] Wrote plots under {PLOTS}/ and markdown {MD}")

if __name__ == "__main__":
    main()
