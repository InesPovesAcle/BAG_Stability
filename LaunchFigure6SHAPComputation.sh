cd /mnt/newStor/paros/paros_WORK/ines/code/BAG_Stability052627

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

LOG="Figure6_OOF_compute_and_aggregate_$(date +%Y%m%d_%H%M%S).log"

nohup python Figure6_OOF_SHAP_pipeline.py \
  --build-script Figure6_build.py \
  --aggregate-script FIgure6_aggregateC.py \
  --mode compute-and-aggregate \
  --main-model imaging_only \
  --n-jobs 4 \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"