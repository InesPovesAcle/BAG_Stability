cd /mnt/newStor/paros/paros_WORK/ines/code/BAG_Stability052627

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

LOG8="Figure8_longitudinal_delta_cBAG_$(date +%Y%m%d_%H%M%S).log"

nohup python Figure8_longitudinal_delta_cBAG.py \
  --feature-set imaging_only \
  --formats png,pdf \
  --min-n 12 \
  > "$LOG8" 2>&1 &

echo "PID=$!"
echo "LOG8=$LOG8"