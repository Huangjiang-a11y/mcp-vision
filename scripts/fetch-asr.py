#!/usr/bin/env python3
"""下载转写模型到 ~/asr（或 MVV_ASR_HOME）。

默认走魔搭（国内直连快）：xiaowangge/sherpa-onnx-sense-voice-small 的 q8 权重，
与 sherpa-onnx 官方 int8 逐字节等价（实测同一段音频输出完全一致）。

VAD 仍取 GitHub release 的 silero_vad.onnx（629 KB）。
注意：魔搭同仓库里那份 silero-vad/model.onnx（2.3 MB）是另一版导出，
喂给 sherpa-onnx 的 VoiceActivityDetector 会一段都切不出来，别用。
"""
import os, sys, urllib.request
from pathlib import Path

MS = "https://www.modelscope.cn/models/xiaowangge/sherpa-onnx-sense-voice-small/resolve/master/"
GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
HOME = Path(os.environ.get("MVV_ASR_HOME", "~/asr")).expanduser()
UA = {"User-Agent": "Mozilla/5.0"}

JOBS = [
    (MS + "tokens.txt",      HOME / "sensevoice" / "tokens.txt",      309 * 1024),
    (MS + "model_q8.onnx",   HOME / "sensevoice" / "model.int8.onnx", 239234116),
    (GH + "silero_vad.onnx", HOME / "vad" / "silero_vad.onnx",        643854),
]


def fetch(url, dest, expect):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expect:
        print("✓ %s 已存在 (%d 字节)" % (dest.name, expect))
        return True
    print("↓ %s" % dest.name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900) as r, \
            open(tmp, "wb") as f:
        total = int(r.headers.get("content-length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done % (4 << 20) < (1 << 20):
                sys.stdout.write("\r   %5.1f%%  %d/%d MB" % (done * 100.0 / total, done >> 20, total >> 20))
                sys.stdout.flush()
    if total:
        sys.stdout.write("\r   %5.1f%%  %d/%d MB\n" % (100.0, total >> 20, total >> 20))
    else:
        print("   %d MB" % (done >> 20))
    if expect and tmp.stat().st_size != expect:
        print("  ⚠️ 大小不符：%d，期望 %d" % (tmp.stat().st_size, expect))
    tmp.replace(dest)
    print("  ✓ %s (%d 字节)" % (dest, dest.stat().st_size))
    return True


if __name__ == "__main__":
    for url, dest, expect in JOBS:
        try:
            fetch(url, dest, expect)
        except Exception as e:
            print("  ✗ %s：%s" % (dest.name, e))
            sys.exit(1)
    print("\n模型目录：%s" % HOME)
