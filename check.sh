#!/usr/bin/env sh
echo "mcp-vision 环境体检"
for c in python3 ffmpeg ffprobe; do
  command -v $c >/dev/null 2>&1 && echo "  [OK] $c" || echo "  [!!] $c 缺失"
done
python3 -c "import numpy; print('  [OK] numpy', numpy.__version__)" 2>/dev/null || echo "  [--] numpy 缺失（audio 不可用）"
python3 -c "import onnxruntime; print('  [OK] onnxruntime', onnxruntime.__version__)" 2>/dev/null || echo "  [--] onnxruntime 缺失（OCR 不可用；pkg install python-onnxruntime）"
python3 -c "import yt_dlp; print('  [OK] yt_dlp')" 2>/dev/null || echo "  [--] yt_dlp 缺失（链接下载不可用）"
if [ -f models/ch_PP-OCRv4_det_infer.onnx ] && [ -f models/ch_PP-OCRv4_rec_infer.onnx ]; then
  echo "  [OK] OCR 模型已就位"
else
  echo "  [--] OCR 模型缺失（python3 scripts/fetch-models.py）"
fi
echo "冒烟：MCP tools/list"
printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n' | python3 mcp_server.py | tail -1 | python3 -c "import sys,json; d=json.load(sys.stdin); print('  工具:', [t['name'] for t in d['result']['tools']]); print('  action:', d['result']['tools'][0]['inputSchema']['properties']['action']['enum'])"
