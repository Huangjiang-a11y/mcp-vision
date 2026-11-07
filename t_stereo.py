"""L/R 分层出图的冒烟测试：合成左 440Hz、右 3000Hz，看两半是否各归各的。"""
import subprocess, sys, tempfile, os, shutil
import engine, audiochart

d = tempfile.mkdtemp(prefix="st-")
st = os.path.join(d, "st.wav")
mo = os.path.join(d, "mo.wav")
subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y",
  "-f","lavfi","-i","sine=frequency=440:duration=3:sample_rate=48000",
  "-f","lavfi","-i","sine=frequency=3000:duration=3:sample_rate=48000",
  "-filter_complex","[0:a]volume=0.5[l];[1:a]volume=0.5[r];[l][r]join=inputs=2:channel_layout=stereo[a]",
  "-map","[a]","-ac","2",st], check=True)
subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y","-i",st,"-ac","1",mo], check=True)

ok = True
r = engine.analyze_spectrum(st, 0, 2, out_dir=d)
fl, fr = r["left"]["spectrum"]["peak_freq_hz"], r["right"]["spectrum"]["peak_freq_hz"]
print("stereo: mono=%s L=%.1fHz R=%.1fHz" % (r["mono"], fl, fr))
if not (abs(fl-440) < 30 and abs(fr-3000) < 30):
    print("  FAIL: 左右串了"); ok = False

m = engine.analyze_spectrum(mo, 0, 2, out_dir=d)
print("mono  : mono=%s L=%.1fHz R=%.1fHz" % (m["mono"], m["left"]["spectrum"]["peak_freq_hz"],
                                            m["right"]["spectrum"]["peak_freq_hz"]))
if not m["mono"]:
    print("  FAIL: mono 没标出来"); ok = False

r["_start"], r["_end"] = 0, 2
paths = audiochart.render(r, ["level", "spectrum", "spectrogram", "peak", "wave"], d)
# 尺寸校验：不依赖 PIL，直接读 PNG 头
import struct
for k, p in paths.items():
    with open(p, "rb") as fh:
        head = fh.read(33)
    w, h = struct.unpack(">II", head[16:24])
    good = (w, h) == (1024, 1024)
    ok &= good
    print("  %-12s %dx%d %s" % (k, w, h, "OK" if good else "FAIL"))

shutil.rmtree(d, ignore_errors=True)
print("RESULT", "ALL OK" if ok else "HAS FAIL")
sys.exit(0 if ok else 1)
