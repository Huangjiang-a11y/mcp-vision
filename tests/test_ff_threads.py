"""engine.py 每条 ffmpeg 命令都要带 -threads N，且必须落在 -i 之前。

为什么较真：ffmpeg 默认 auto 按核数开解码线程，手机上 8 个小核全被抢走，
同时跑的 ONNX/ASR 被挤慢，实测整体更慢。这个开关是靠「每条命令都记得写」
维持的，漏一条就等于没锁（不同工具走不同命令）。
不碰 ffmpeg 也不碰素材：把 subprocess.run 换成一个记账的假函数。
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine

calls = []


class _Fake:
    returncode = 0
    # 给点非空 stdout：pcm16k 拿不到数据会主动报错（那是它的职责）
    stdout = b"\0" * 32
    stderr = ""


def fake_run(args, **kw):
    calls.append(list(args))
    f = _Fake()
    if "-filter_complex" in args:
        # 批量缩略图那条要 36x32xN 字节才肯往下走，不然它会退回逐张、多记 N 条命令
        f.stdout = b"\0" * (36 * 32 * args.count("-i"))
    return f


engine.subprocess.run = fake_run
engine.audio_info = lambda media: {"sample_rate": 16000, "channels": 2}

fails = []
ncheck = 0


def check(name, ok, detail=""):
    global ncheck
    ncheck += 1
    if ok:
        print("  ok   %s" % name)
    else:
        fails.append("%s %s" % (name, detail))
        print("  FAIL %s %s" % (name, detail))


with tempfile.TemporaryDirectory() as td:
    media = os.path.join(td, "x.mp4")
    open(media, "wb").write(b"\0" * 16)
    # analyze_spectrum 会去读自己刚「生成」的 pcm，先把假数据放好：16000 字节 = 0.5 秒 16k 交错立体声
    pcm = os.path.join(td, "s-0-1000.pcm")
    open(pcm, "wb").write(b"\0" * 16000)

    engine.scene_scores(media, 0, 10)
    engine._extract_jpeg(media, 1.0, 640, os.path.join(td, "f.jpg"))
    engine._load_raw_rgb(media, 1.0, 8, 8)
    try:
        engine.analyze_spectrum(media, 0, 1.0, out_dir=td)
    except Exception as e:  # 只要命令记下来了就算过，后面的数值计算不关心
        print("  note analyze_spectrum 后续计算: %s" % type(e).__name__)
    try:
        engine.pcm16k(media, 0, 1.0)
    except Exception as e:
        check("pcm16k 拿得到音频", False, repr(e))

    # 逐帧特征（_gray_thumb）与批量缩略图（hstack）也是解码，同样要锁
    jpgs = [os.path.join(td, "a.jpg"), os.path.join(td, "b.jpg")]
    for j in jpgs:
        open(j, "wb").write(b"\0" * 16)
    engine.frame_features(jpgs[0])
    engine.frame_features_many(jpgs)

ff = [c for c in calls if c and c[0] == "ffmpeg"]
check("七条 ffmpeg 命令都记到了", len(ff) == 7, "实际 %d 条" % len(ff))

want = engine.FF_THREADS


def per_input_ok(c):
    """每个 -i 前面都得有自己的 -threads：批量那条有 36 个输入，
    只在开头写一个是只管第一个输入的。"""
    idx = [i for i, x in enumerate(c) if x == "-i"]
    ok = True
    for k, i in enumerate(idx):
        seg = c[(idx[k - 1] if k else 0):i]
        if "-threads" not in seg or seg[seg.index("-threads") + 1] != str(want):
            ok = False
    return ok


for c in ff:
    tag = " ".join(x for x in c[1:] if not x.startswith("-"))[:60]
    has = "-threads" in c and c[c.index("-threads") + 1] == str(want)
    check("-threads %d: %s" % (want, tag), has, "命令 %r" % (c[:8],))
    check("每个 -i 前都有 -threads: %s" % tag, per_input_ok(c),
          "%d 个输入" % c.count("-i"))

check("工具那条默认 3 条", want == 3, "实际 %r" % (want,))
ocr = __import__("ocr")
# 两条现在分开了：工具同一时间只跑一路给 3；字幕默认 2 进程 × 2 条
check("字幕那条默认 2 条", ocr.FFMPEG_THREADS == 2, "实际 %r" % (ocr.FFMPEG_THREADS,))
import inspect as _i2
check("工具读 MVV_FFMPEG_THREADS", "MVV_FFMPEG_THREADS" in _i2.getsource(engine)[:2000])
check("字幕读 MVV_OCR_FFMPEG_THREADS", "MVV_OCR_FFMPEG_THREADS" in _i2.getsource(ocr)[:2000])
# ocr.py 自己那条命令不能写死 2：定义了 FFMPEG_THREADS 却不用，等于环境变量是假的
import inspect as _insp
src = _insp.getsource(ocr._stream_frames)
check("_stream_frames 用常量", "str(FFMPEG_THREADS)" in src)
check("_stream_frames 没写死 2", '"-threads", "2"' not in src)
# 兜底那条（逐点 seek 取采样帧）以前漏了锁，批量缩略图也是
src2 = _insp.getsource(ocr._collect_sample_frames)
check("取采样帧的兜底也锁了", '"-threads", str(FFMPEG_THREADS)' in src2)

print("ff_threads: %d 项检查，%d 项失败" % (ncheck, len(fails)))
sys.exit(1 if fails else 0)
