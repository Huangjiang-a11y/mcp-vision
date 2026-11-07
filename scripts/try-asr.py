import subprocess, wave, numpy as np, sherpa_onnx
V = "out/real/BV1Rx54znE7k_p1.mp4"
rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model="/data/data/com.termux/files/home/asr/sensevoice/model.int8.onnx",
    tokens="/data/data/com.termux/files/home/asr/sensevoice/tokens.txt",
    num_threads=2, use_itn=True, language="zh")

def run(a, b, af, label):
    cmd = ["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y",
           "-ss",str(a),"-t",str(b-a),"-i",V,"-vn","-af",af,
           "-ac","1","-ar","16000","-f","wav","-"]
    p = subprocess.run(cmd, capture_output=True, timeout=120)
    if p.returncode: return "(ffmpeg 失败)"
    import io
    w = wave.open(io.BytesIO(p.stdout), "rb")
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
    s = rec.create_stream(); s.accept_waveform(16000, x); rec.decode_stream(s)
    return s.result.text

A, B = 72.0, 80.0
tests = [
    ("原始",                         "anull"),
    ("0.8x + 1.2x音量",              "atempo=0.8,volume=1.2"),
    ("0.8x + 1.5x音量 + 带通200-5k",  "atempo=0.8,volume=1.5,highpass=f=200,lowpass=f=5000"),
    ("0.7x + 1.8x音量 + 带通150-5k",  "atempo=0.7,volume=1.8,highpass=f=150,lowpass=f=5000"),
    ("0.7x + 2x音量 + 带通150-4k + 降噪","atempo=0.7,volume=2.0,highpass=f=150,lowpass=f=4000,afftdn=nf=-25"),
]
for label, af in tests:
    print("%-34s %s" % (label, run(A, B, af, label)))
