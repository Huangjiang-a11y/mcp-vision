#!/usr/bin/env sh
echo "mcp-video-vision 环境体检"
for c in python3 ffmpeg ffprobe; do
  if command -v $c >/dev/null 2>&1; then echo "  [OK] $c"; else echo "  [!!] $c 缺失"; fi
done
python3 -c "import numpy; print('  [OK] numpy', numpy.__version__)" 2>/dev/null || echo "  [--] numpy 缺失（audio 信号分析不可用）"
python3 -c "import sherpa_onnx; print('  [OK] sherpa_onnx')" 2>/dev/null || echo "  [--] sherpa_onnx 缺失（transcribe 不可用）"
python3 -c "import yt_dlp; print('  [OK] yt_dlp')" 2>/dev/null || echo "  [--] yt_dlp 缺失（链接下载不可用）"
echo "冒烟：MCP tools/list"
printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n' | python3 mcp_server.py | tail -1 | python3 -c "import sys,json; d=json.load(sys.stdin); print('  工具:', [t['name'] for t in d['result']['tools']])"
