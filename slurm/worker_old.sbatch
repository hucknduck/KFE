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

set -euo pipefail

CMD_FILE="${1:-}"
if [[ -z "$CMD_FILE" || ! -f "$CMD_FILE" ]]; then
  echo "Usage: $0 <batch_cmds_file>" >&2
  exit 2
fi

mkdir -p logs results/records

job_ids=()
line_no=0
while IFS= read -r line || [[ -n "$line" ]]; do
    line_no=$((line_no+1))
    # skip empty or commented lines
    [[ -z "${line// }" ]] && continue
    [[ "$line" =~ ^# ]] && continue

    echo "[worker] Submitting ($line_no): $line"

    # run the sbatch line and capture output (e.g., "Submitted batch job 12345")
    set +e
    out=$(bash -lc "$line" 2>&1)
    rc=$?
    set -e
    echo "$out"

    if [[ $rc -ne 0 ]]; then
        echo "[worker] ERROR submitting line $line_no (rc=$rc), continuing..."
        continue
    fi

    if [[ "$out" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
        jid="${BASH_REMATCH[1]}"
        job_ids+=("$jid")
        echo "[worker] -> leaf job id: $jid"
    else
        echo "[worker] WARNING: could not parse job id for line $line_no"
    fi

    # small throttle to avoid hammering the scheduler
    sleep 1
done < "$CMD_FILE"

if [[ ${#job_ids[@]} -eq 0 ]]; then
    echo "[worker] No jobs submitted from $CMD_FILE"
    exit 0
fi

echo "[worker] Waiting for ${#job_ids[@]} leaf jobs to finish..."
list=$(IFS=,; echo "${job_ids[*]}")

# Poll until none of the leaf job IDs remain in the queue
while true; do
    set +e
    remaining=$(squeue -h -j "$list" -o "%i" 2>/dev/null | wc -l)
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "[worker] WARNING: squeue failed, assuming 0 jobs remain"
        remaining=0
    fi

    if [[ "$remaining" -eq 0 ]]; then
        break
    fi
    echo "[worker] $remaining leaf jobs still running... sleeping 60s"
    sleep 60
done

echo "[worker] All leaf jobs finished."
