#!/data/data/com.termux/files/usr/bin/bash
cd /data/data/com.termux/files/home/mcp-vision
pkill -f "[m]cp_server.py"
sleep 1
nohup python3 mcp_server.py --http 8765 > server.log 2>&1 &
sleep 3
pgrep -af "[m]cp_server.py"
