#!/usr/bin/env bash
set -euo pipefail

# Run ADNI-only strict longitudinal Figure 8 with hippocampal volume as percent brain in the main figure.
# Place/run this script from:
#   /mnt/newStor/paros/paros_WORK/ines/code/BAG_Stability052627
#
# It expects:
#   Figure8_ADNI_longitudinal_delta_cBAG_STRICT.py
# to be in the same directory.

cd /mnt/newStor/paros/paros_WORK/ines/code/BAG_Stability052627

if [[ ! -f "Figure8_ADNI_longitudinal_delta_cBAG_STRICT.py" ]]; then
  echo "ERROR: Figure8_ADNI_longitudinal_delta_cBAG_STRICT.py not found in:"
  pwd
  echo
  echo "Please copy/download it into this directory first."
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

OUTDIR="/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure8_ADNI_longitudinal_delta_cBAG_STRICT_pctHc"
LOG8="Figure8_ADNI_STRICT_pctHc_$(date +%Y%m%d_%H%M%S).log"

echo "Launching ADNI-only strict Figure 8 with Hc volume (% brain) in main Panel C..."
echo "Working directory: $(pwd)"
echo "Output directory: ${OUTDIR}"
echo "Log: ${LOG8}"

nohup python Figure8_ADNI_longitudinal_delta_cBAG_STRICT.py \
  --feature-set imaging_only \
  --formats png,pdf \
  --min-n 12 \
  --main-endpoints "Cognition composite,Hippocampal volume (% brain),Hippocampal FA,Total brain volume,Total brain FA,Global efficiency" \
  --outdir "${OUTDIR}" \
  > "${LOG8}" 2>&1 &

PID=$!

echo "PID=${PID}"
echo "LOG8=${LOG8}"
echo
echo "Monitor with:"
echo "  tail -f ${LOG8}"
echo
echo "Check outputs with:"
echo "  find ${OUTDIR} -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort"
