#!/usr/bin/env bash
#SBATCH --output=logs_new/worker_%j.out
#SBATCH --error=logs_new/worker_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=kfe_eval_worker
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
###SBATCH --constraint=geforce_rtx_3090

nvidia-smi
echo "CUDA_VISIBLE_DEVICES is $CUDA_VISIBLE_DEVICES"


set -euo pipefail

CMD_FILE="${1:-}"
if [[ -z "$CMD_FILE" || ! -f "$CMD_FILE" ]]; then
  echo "Usage: $0 <batch_cmds_file>" >&2
  exit 2
fi

mkdir -p logs results/records

line_no=0
while IFS= read -r line || [[ -n "$line" ]]; do
    line_no=$((line_no+1))
    # skip empty or commented lines
    [[ -z "${line// }" ]] && continue
    [[ "$line" =~ ^# ]] && continue

    # replace --device value in the command with the allocated GPU
    # line=$(echo "$line" | sed -E "s/--device [0-9]+/--device $GPU/")

    echo "[worker] Running ($line_no): $line"

    # Run the command and capture output
    set +e
    out=$($line 2>&1)
    rc=$?
    set -e
    echo "$out"

    if [[ $rc -ne 0 ]]; then
        echo "[worker] ERROR: command $line_no failed (rc=$rc), continuing..."
    fi
done < "$CMD_FILE"

echo "[worker] All commands finished."