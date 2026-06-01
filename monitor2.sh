watch -n 10 '
echo "===== LOAD ====="
uptime
echo
echo "===== MEMORY/SWAP ====="
free -h
echo
echo "===== SWAP ACTIVITY ====="
vmstat 1 2 | tail -1
echo
echo "===== YOUR TOP PROCESSES ====="
ps -u $USER -o pid,ppid,etime,%cpu,%mem,cmd --sort=-%cpu | head -20
echo
echo "===== VALIDATION PROCESSES ====="
pgrep -af "4_validation_BOTH_OOF_AND_FULLCOHORT|validation_both_oof_fullcohort" || true
echo
echo "===== TEMPS ====="
sensors | grep -E "Tctl|Tccd" | head -20
'