import subprocess, sys, tempfile, os, shutil
import engine

d = tempfile.mkdtemp(prefix="ch-")
st = os.path.join(d, "st.wav")
mo = os.path.join(d, "mo.wav")
# 左 440Hz，右 3000Hz，各 -6dB
subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y",
  "-f","lavfi","-i","sine=frequency=440:duration=3:sample_rate=48000",
  "-f","lavfi","-i","sine=frequency=3000:duration=3:sample_rate=48000",
  "-filter_complex","[0:a]volume=0.5[l];[1:a]volume=0.5[r];[l][r]join=inputs=2:channel_layout=stereo[a]",
  "-map","[a]","-ac","2",st], check=True)
subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y","-i",st,"-ac","1",mo], check=True)

def run(f, ch):
    r = engine.analyze_spectrum(f, 0, 2, out_dir=d, channel=ch)
    return r["channel"], r["channels"], r["spectrum"]["peak_freq_hz"], r["level"]["peak_dbfs"]

ok = True
for ch, want in (("left",440),("right",3000)):
    c, n, f, pk = run(st, ch)
    good = abs(f-want) < 30
    ok &= good
    print("%-5s ch=%d peak=%7.1fHz 期望%5d  %s" % (ch, n, f, want, "OK" if good else "FAIL"))
c, n, f, pk = run(st, "mix")
print("mix   ch=%d peak=%7.1fHz (左右都在，峰值取到其中一个即可) db=%.1f" % (n, f, pk))
print("默认 channel 参数:", engine.analyze_spectrum(st, 0, 1, out_dir=d)["channel"])

try:
    engine.analyze_spectrum(mo, 0, 2, out_dir=d, channel="left")
    print("单声道传 left: FAIL（没报错）"); ok = False
except RuntimeError as e:
    print("单声道传 left: OK ->", e)
try:
    engine.analyze_spectrum(st, 0, 2, out_dir=d, channel="both")
    print("非法 channel: FAIL"); ok = False
except RuntimeError as e:
    print("非法 channel: OK ->", e)

shutil.rmtree(d, ignore_errors=True)
print("RESULT", "ALL OK" if ok else "HAS FAIL")
sys.exit(0 if ok else 1)
