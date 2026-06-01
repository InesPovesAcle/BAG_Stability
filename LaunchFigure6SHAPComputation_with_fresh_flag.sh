#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Launch Figure 6 SHAP computation / aggregation
# =============================================================================
#
# Default behavior:
#   Recompute SHAP + aggregate using existing OUT folder.
#
# Fresh recomputation:
#   Use --fresh to move the existing Figure6_SHAP folder to a timestamped backup
#   before running. This prevents Figure6_build.py from reusing existing
#   subject-level SHAP CSVs.
#
# Examples:
#
#   Full recomputation, clean output:
#     bash LaunchFigure6SHAPComputation_with_fresh_flag.sh --fresh
#
#   Reduced smoke test:
#     bash LaunchFigure6SHAPComputation_with_fresh_flag.sh \
#       --fresh \
#       --cohorts ADNI \
#       --feature-sets imaging_only \
#       --max-subjects 10 \
#       --n-jobs 2 \
#       --formats png \
#       --dpi 150
#
#   Global-only faster run:
#     bash LaunchFigure6SHAPComputation_with_fresh_flag.sh \
#       --fresh \
#       --run-node-shap 0 \
#       --run-edge-shap 0
#
# =============================================================================

WORK="${WORK:-/mnt/newStor/paros/paros_WORK}"
CODE_DIR="$WORK/ines/code/BAG_Stability052627"
OUT="$WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure6_SHAP"

BUILD_SCRIPT="Figure6_build.py"
AGG_SCRIPT="FIgure6_aggregateC.py"
PIPELINE_SCRIPT="Figure6_OOF_SHAP_pipeline.py"

MODE="compute-and-aggregate"
MAIN_MODEL="imaging_only"
COHORTS="ADNI,ADRC,HABS,AD_DECODE"
FEATURE_SETS="imaging_only,imaging_demographics,imaging_biomarkers,full,full_no_cardiovascular"
N_JOBS="4"
MAX_SUBJECTS="None"
RUN_GLOBAL_SHAP="1"
RUN_NODE_SHAP="1"
RUN_EDGE_SHAP="1"
FORMATS="png,pdf"
DPI="450"
FRESH="0"
DRY_RUN="0"

usage() {
  cat <<EOF
Usage:
  bash $0 [options]

Options:
  --fresh
      Move existing Figure6_SHAP OUT folder to OUT_backup_<timestamp> before running.

  --mode MODE
      compute-and-aggregate, aggregate-only, or aggregate-master-only.
      Default: ${MODE}

  --cohorts LIST
      Comma-separated cohorts.
      Default: ${COHORTS}

  --feature-sets LIST
      Comma-separated feature sets.
      Default: ${FEATURE_SETS}

  --main-model MODEL
      Default: ${MAIN_MODEL}

  --n-jobs N
      Joblib workers for SHAP recomputation.
      Default: ${N_JOBS}

  --max-subjects N|None
      Limit subjects for tests.
      Default: ${MAX_SUBJECTS}

  --run-global-shap 0|1
      Default: ${RUN_GLOBAL_SHAP}

  --run-node-shap 0|1
      Default: ${RUN_NODE_SHAP}

  --run-edge-shap 0|1
      Default: ${RUN_EDGE_SHAP}

  --formats LIST
      Example: png or png,pdf.
      Default: ${FORMATS}

  --dpi N
      Default: ${DPI}

  --dry-run
      Print commands but do not run.

Examples:
  bash $0 --fresh

  bash $0 --fresh --cohorts ADNI --feature-sets imaging_only --max-subjects 10 --n-jobs 2 --formats png --dpi 150

  bash $0 --fresh --run-node-shap 0 --run-edge-shap 0 --formats png --dpi 250
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)
      FRESH="1"
      shift
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --cohorts)
      COHORTS="$2"
      shift 2
      ;;
    --feature-sets|--models)
      FEATURE_SETS="$2"
      shift 2
      ;;
    --main-model)
      MAIN_MODEL="$2"
      shift 2
      ;;
    --n-jobs)
      N_JOBS="$2"
      shift 2
      ;;
    --max-subjects)
      MAX_SUBJECTS="$2"
      shift 2
      ;;
    --run-global-shap)
      RUN_GLOBAL_SHAP="$2"
      shift 2
      ;;
    --run-node-shap)
      RUN_NODE_SHAP="$2"
      shift 2
      ;;
    --run-edge-shap)
      RUN_EDGE_SHAP="$2"
      shift 2
      ;;
    --formats)
      FORMATS="$2"
      shift 2
      ;;
    --dpi)
      DPI="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$CODE_DIR" ]]; then
  echo "[ERROR] Code directory not found: $CODE_DIR"
  exit 1
fi

cd "$CODE_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

echo "================================================================================"
echo "Figure 6 SHAP launcher"
echo "================================================================================"
echo "WORK:          $WORK"
echo "CODE_DIR:      $CODE_DIR"
echo "OUT:           $OUT"
echo "MODE:          $MODE"
echo "COHORTS:       $COHORTS"
echo "FEATURE_SETS:  $FEATURE_SETS"
echo "MAIN_MODEL:    $MAIN_MODEL"
echo "N_JOBS:        $N_JOBS"
echo "MAX_SUBJECTS:  $MAX_SUBJECTS"
echo "GLOBAL SHAP:   $RUN_GLOBAL_SHAP"
echo "NODE SHAP:     $RUN_NODE_SHAP"
echo "EDGE SHAP:     $RUN_EDGE_SHAP"
echo "FORMATS:       $FORMATS"
echo "DPI:           $DPI"
echo "FRESH:         $FRESH"
echo "DRY_RUN:       $DRY_RUN"
echo "================================================================================"

if [[ "$FRESH" == "1" ]]; then
  if [[ "$MODE" != "compute-and-aggregate" ]]; then
    echo "[WARN] --fresh was requested, but mode is '$MODE'."
    echo "[WARN] Fresh is mainly useful with --mode compute-and-aggregate."
  fi

  if [[ -e "$OUT" ]]; then
    BACKUP="${OUT}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "[FRESH] Moving existing OUT to backup:"
    echo "        $OUT"
    echo "    ->  $BACKUP"

    if [[ "$DRY_RUN" == "0" ]]; then
      mv "$OUT" "$BACKUP"
    fi
  else
    echo "[FRESH] OUT does not exist yet; no backup needed."
  fi

  if [[ "$DRY_RUN" == "0" ]]; then
    mkdir -p "$OUT"
  fi
fi

LOG="Figure6_OOF_${MODE}_$(date +%Y%m%d_%H%M%S).log"

CMD=(
  python "$PIPELINE_SCRIPT"
  --build-script "$BUILD_SCRIPT"
  --aggregate-script "$AGG_SCRIPT"
  --mode "$MODE"
  --main-model "$MAIN_MODEL"
  --cohorts "$COHORTS"
  --feature-sets "$FEATURE_SETS"
  --n-jobs "$N_JOBS"
  --max-subjects "$MAX_SUBJECTS"
  --run-global-shap "$RUN_GLOBAL_SHAP"
  --run-node-shap "$RUN_NODE_SHAP"
  --run-edge-shap "$RUN_EDGE_SHAP"
  --formats "$FORMATS"
  --dpi "$DPI"
)

echo
echo "Command:"
printf ' %q' "${CMD[@]}"
echo
echo
echo "LOG: $LOG"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY RUN] Not launching."
  exit 0
fi

nohup "${CMD[@]}" > "$LOG" 2>&1 &

PID=$!

echo "PID=$PID"
echo "LOG=$LOG"
echo
echo "Monitor with:"
echo "  tail -f $CODE_DIR/$LOG"
echo
echo "Check process:"
echo "  ps -p $PID -o pid,ppid,etime,%cpu,%mem,cmd"
echo
echo "Check output counts:"
echo "  OUT=\"$OUT\""
echo "  find \"\$OUT\" -type f -name 'global_feature_shap_subject_*.csv' | wc -l"
echo "  find \"\$OUT\" -type f -name 'node_feature_shap_subject_*.csv' | wc -l"
echo "  find \"\$OUT\" -type f -name 'edge_shap_subject_*.csv' | wc -l"
