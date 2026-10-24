#!/usr/bin/env bash
#SBATCH --array=1-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h200:1
set -euo pipefail
cd -- /shared/scan
: "${SLURM_ARRAY_TASK_ID:?set the Slurm array task ID}"
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > 1 )); then exit 2; fi
pids=()
for (( instance=0; instance<1; instance++ )); do
  (
    export SOLPROP_SLURM_WORKER=1
    export SOLPROP_SCAN_INSTANCE="$instance"
    export SOLPROP_SCAN_CPU_OFFSET=$(( instance * 8 ))
    export SOLPROP_SCAN_CPU_COUNT=8
    export NUMBA_NUM_THREADS=2
    export SOLPROP_THREADS=2
    export OMP_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export VECLIB_MAXIMUM_THREADS=2
    export OMP_WAIT_POLICY=passive
    exec /python -m solprop.scan /shared/scan/single.py --queue '/shared/scan/single queue' --threads 2 --memory-limit 4294967296 --workers 1
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
