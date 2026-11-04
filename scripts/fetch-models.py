#!/usr/bin/env python3
"""下载 PP-OCRv4 模型（约 16MB）到 models/。需要时设 https_proxy。"""
import urllib.request, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "models"
DEST.mkdir(parents=True, exist_ok=True)
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
FILES = [
 ("ch_PP-OCRv4_det_infer.onnx", "%s/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx" % HF, 4745517),
 ("ch_PP-OCRv4_rec_infer.onnx", "%s/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx" % HF, 10857958),
 ("ppocr_keys_v1.txt", "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt", 26250),
]
for name, url, size in FILES:
    d = DEST / name
    if d.exists() and d.stat().st_size == size:
        print("✓ %s 已存在" % name); continue
    print("下载 %s ..." % name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=300).read()
        d.write_bytes(data)
        print("  ✓ %d 字节%s" % (len(data), "" if len(data) == size else "  ⚠️ 期望 %d" % size))
    except Exception as e:
        print("  ✗ %s: %s" % (type(e).__name__, e)); sys.exit(1)
print("模型目录:", DEST)
