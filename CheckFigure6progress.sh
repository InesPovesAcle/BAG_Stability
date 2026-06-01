cd /mnt/newStor/paros/paros_WORK/ines/code/BAG_Stability052627

OUT=/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure6_SHAP
LOG=$(ls -t Figure6_OOF_compute-and-aggregate_*.log 2>/dev/null | head -1)

watch -n 60 '
echo "===== TIME ====="
date
echo

echo "===== LOAD ====="
uptime
echo

echo "===== MEMORY / SWAP ====="
free -h
echo

echo "===== FIGURE 6 PROCESSES ====="
ps -u ines -o pid,ppid,stat,etime,%cpu,%mem,rss,cmd --sort=-%cpu \
  | grep -E "Figure6|LokyProcess|joblib|python" \
  | head -25
echo

echo "===== SHAP SUBJECT FILE COUNTS ====="
echo "Global:" $(find "'"$OUT"'" -type f -name "global_feature_shap_subject_*.csv" | wc -l)
echo "Node:  " $(find "'"$OUT"'" -type f -name "node_feature_shap_subject_*.csv" | wc -l)
echo "Edge:  " $(find "'"$OUT"'" -type f -name "edge_shap_subject_*.csv" | wc -l)
echo

echo "===== COMPLETED COHORT/MODEL SUMMARIES ====="
find "'"$OUT"'" -type f -name "shap_run_summary.csv" \
  -printf "%TY-%Tm-%Td %TH:%TM %p\n" | sort
echo

echo "===== NEWEST SHAP FILES ====="
find "'"$OUT"'" -type f \
  \( -name "global_feature_shap_subject_*.csv" \
  -o -name "node_feature_shap_subject_*.csv" \
  -o -name "edge_shap_subject_*.csv" \) \
  -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" \
  | sort | tail -10
echo

echo "===== LATEST LOG ====="
echo "'"$LOG"'"
tail -40 "'"$LOG"'"
'