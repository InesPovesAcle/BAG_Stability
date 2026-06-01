echo "===== HOST / TIME ====="
hostname
date

echo "===== LOAD / CPU ====="
uptime
nproc
top -bn1 | head -20

echo "===== MEMORY ====="
free -h

echo "===== DISK SPACE ====="
df -h /mnt/newStor /tmp . 2>/dev/null

echo "===== DISK I/O ====="
iostat -xz 1 3 2>/dev/null || echo "iostat not available"

echo "===== GPU ====="
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU visible"

echo "===== CPU TEMPERATURE ====="
sensors 2>/dev/null || echo "sensors not available"