#!/usr/bin/env bash
#SBATCH --array=1-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --partition='cpu queue'
#SBATCH --gres=gpu:h200:2
set -euo pipefail
cd -- /shared/scan
: "${SLURM_ARRAY_TASK_ID:?set the Slurm array task ID}"
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > 3 )); then exit 2; fi
IFS=, read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
if (( ${#devices[@]} < 2 )) || [[ -z "${devices[0]}" ]]; then echo "insufficient assigned GPUs" >&2; exit 2; fi
pids=()
for (( instance=0; instance<2; instance++ )); do
  (
    export SOLPROP_SLURM_WORKER=1
    export SOLPROP_SCAN_INSTANCE="$instance"
    export SOLPROP_SCAN_CPU_OFFSET=$(( instance * 4 ))
    export SOLPROP_SCAN_CPU_COUNT=4
    export CUDA_VISIBLE_DEVICES="${devices[$instance]}"
    export NUMBA_NUM_THREADS=2
    export OMP_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export VECLIB_MAXIMUM_THREADS=2
    export OMP_WAIT_POLICY=passive
    exec /python -m solprop.scan '/shared/scan/scan '"'"'quoted'"'"'.py' --queue '/shared/scan/point queue' --threads 2 --memory-limit 6442450944 --workers 1 --fresh --chunk "$(( (SLURM_ARRAY_TASK_ID - 1) * 2 + instance ))/6"
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
