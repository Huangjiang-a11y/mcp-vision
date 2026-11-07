#!/usr/bin/env sh
echo "mcp-vision 环境体检"
[ -f ./env.sh ] && . ./env.sh
for c in python3 ffmpeg ffprobe; do
  command -v $c >/dev/null 2>&1 && echo "  [OK] $c" || echo "  [!!] $c 缺失"
done
python3 -c "import numpy; print('  [OK] numpy', numpy.__version__)" 2>/dev/null || echo "  [--] numpy 缺失（audio 不可用）"
python3 -c "import onnxruntime; print('  [OK] onnxruntime', onnxruntime.__version__)" 2>/dev/null || echo "  [--] onnxruntime 缺失（OCR 不可用；pkg install python-onnxruntime）"
python3 -c "import yt_dlp; print('  [OK] yt_dlp')" 2>/dev/null || echo "  [--] yt_dlp 缺失（链接下载不可用）"
python3 -c "import sherpa_onnx; print('  [OK] sherpa_onnx')" 2>/dev/null || echo "  [--] sherpa_onnx 缺失（transcribe 不可用）"

miss=""
for v in MVV_ASR_MODEL MVV_ASR_TOKENS MVV_VAD_MODEL; do
  eval "p=\$$v"
  [ -n "$p" ] && [ -f "$p" ] || miss="$miss $v"
done
if [ -z "$miss" ]; then
  echo "  [OK] STT 模型（MVV_ASR_MODEL / MVV_ASR_TOKENS / MVV_VAD_MODEL）"
else
  echo "  [--] STT 模型未配:$miss（见 INSTALL.md 3.2）"
fi

det="${MVV_OCR_DET:-models/ch_PP-OCRv4_det_infer.onnx}"
rec="${MVV_OCR_REC:-models/ch_PP-OCRv4_rec_infer.onnx}"
if [ -f "$det" ] && [ -f "$rec" ]; then
  echo "  [OK] OCR 模型"
else
  echo "  [--] OCR 模型缺失（设 MVV_OCR_DET / MVV_OCR_REC / MVV_OCR_KEYS）"
fi
echo "冒烟：MCP tools/list"
printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n' | python3 mcp_server.py | tail -1 | python3 -c "import sys,json; d=json.load(sys.stdin); print('  工具:', [t['name'] for t in d['result']['tools']]); print('  action:', d['result']['tools'][0]['inputSchema']['properties']['action']['enum'])"

# --test：跑不需要模型、不需要素材的回归测试
if [ "$1" = "--test" ]; then
  echo "测试：L/R 分层出图"
  python3 tests/test_stereo.py | sed 's/^/  /' || exit 1
  echo "测试：取窗与参数校验"
  python3 tests/test_windows.py | sed 's/^/  /' || exit 1
fi
