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
# 测试跑挂了必须让整个脚本失败。
# 原来写的是 `python3 x.py | sed ... || exit 1`：管道的退出码取最后一个命令，
# sed 正常退出就是 0，于是 Python 报错/断言失败全被吞掉，CI 一直是假绿。
# （真事：加了一条引用了不存在属性的测试，CI 照样 ✓。）
run_test() {
  _t=$(mktemp)
  python3 "$1" >"$_t" 2>&1
  _rc=$?
  sed 's/^/  /' "$_t"
  rm -f "$_t"
  [ "$_rc" -eq 0 ] || exit "$_rc"
}

if [ "$1" = "--test" ]; then
  echo "测试：L/R 分层出图"
  run_test tests/test_stereo.py
  echo "测试：取窗与参数校验"
  run_test tests/test_windows.py
  echo "测试：整页文字判定"
  run_test tests/test_page_like.py
  echo "测试：字幕行筛选"
  run_test tests/test_keep_wide.py
  echo "测试：ffmpeg 线程数"
  run_test tests/test_ff_threads.py
  echo "测试：选帧筛子"
  run_test tests/test_pick_frames.py
fi
