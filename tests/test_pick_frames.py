"""选帧三道筛子（分桶 C / 黑帧 A / 去重 B）的回归测试。

不碰 ffmpeg 也不碰素材：把 engine.extract_frames / engine.frame_features
换成假的，直接喂「哪几帧是黑的、哪几帧长得一样」。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine, mcp_server as M

fails = []
ncheck = 0


def check(name, got, want):
    global ncheck
    ncheck += 1
    if got != want:
        fails.append("%s\n    期望 %r\n    实际 %r" % (name, want, got))
        print("  FAIL %s" % name)
    else:
        print("  ok   %s" % name)


# --- C：分桶 ---
# 20 秒窗口 4 桶，桶内分数乱序，看是否按分数降序取、跨桶不串
scores = [(1.0, 0.9), (1.5, 0.5), (6.0, 0.2), (16.0, 0.3), (11.0, 0.4)]
buckets, filled = M._scene_buckets(scores, 4, 0, 20)
check("分桶按时间归位", [len(b) for b in buckets], [2, 1, 1, 1])
check("桶内按分数降序", buckets[0], [1.0, 1.5])
check("没有空桶就不补", filled, 0)

# 桶里全是噪声：相对下限（最高分 5%）把它们滤掉，空桶取桶中点
noisy = [(1.0, 0.5), (1.2, 0.01), (11.0, 0.02)]
buckets, filled = M._scene_buckets(noisy, 2, 0, 20)
check("噪声不算变化点", buckets, [[1.0], [15.0]])
check("空桶计一次", filled, 1)

# k 上限：一桶里 5 个变化点最多留 3 个候选
crowd = [(5.0 + i * 0.1, 0.9 - i * 0.1) for i in range(5)]
buckets, _ = M._scene_buckets(crowd, 1, 0, 20)
check("桶内候选数封顶", len(buckets[0]), M.BUCKET_K)
check("桶内候选按分数排", buckets[0][0], 5.0)

check("hamming 数位", (M._hamming(0, 0b1011), M._hamming(0, 0)), (3, 0))

# --- A+B：抽帧后筛 ---
FRAMES = {}     # 帧文件 -> (亮度均值, 标准差, dHash)
AVAILABLE = set()


def fake_extract(media, times, **kw):
    out = []
    for t in times:
        t = round(float(t), 3)
        if t in AVAILABLE:
            out.append({"seconds": t, "path": "f%09d.jpg" % round(t * 1000)})
    return out


def fake_feat(path, timeout=30):
    return FRAMES.get(path, (None, None, None))


def fake_feat_many(paths, timeout=None):
    return [fake_feat(p) for p in paths]


engine.extract_frames = fake_extract
engine.frame_features = fake_feat
engine.frame_features_many = fake_feat_many


def frame(t, mean=90.0, sd=60.0, h=None):
    t = round(float(t), 3)
    AVAILABLE.add(t)
    FRAMES["f%09d.jpg" % round(t * 1000)] = (mean, sd, h)


asset = {"media_path": "x.mp4"}
NEAR = (1 << M.DEDUP_BITS) - 1          # 与 0 差 DEDUP_BITS 位：算同一镜头
FAR = (1 << (M.DEDUP_BITS + 1)) - 1     # 差 DEDUP_BITS+1 位：算换了镜头

# 全正常：一格不换（三个哈希两两差 32 位以上）
FULL, ZERO, ALT = 0xFFFFFFFFFFFFFFFF, 0, 0x5555555555555555
for t, h in ((1.0, FULL), (5.0, ZERO), (9.0, ALT)):
    frame(t, h=h)
frames, stats = M._pick_frames(asset, [[1.0], [5.0], [9.0]], ".", 10.0, (341, 192), None)
check("正常帧原样通过", [f["seconds"] for f in frames], [1.0, 5.0, 9.0])
check("正常帧不记换人", stats, {})

# 黑帧：桶 0 首选 mean=5 被换掉，拿桶内次优
FRAMES.clear(); AVAILABLE.clear()
frame(1.0, mean=5.0, h=FAR); frame(2.0, mean=88.0, h=FAR)
frames, stats = M._pick_frames(asset, [[1.0, 2.0]], ".", 10.0, (341, 192), None)
check("黑帧换桶内次优", [f["seconds"] for f in frames], [2.0])
check("换人原因记 dark", (stats.get("moved"), stats.get("why_dark")), (1, 1))

# 纯色：std 太低
FRAMES.clear(); AVAILABLE.clear()
frame(3.0, sd=1.0, h=FAR); frame(4.0, sd=50.0, h=FAR)
frames, stats = M._pick_frames(asset, [[3.0, 4.0]], ".", 10.0, (341, 192), None)
check("纯色换桶内次优", [f["seconds"] for f in frames], [4.0])
check("换人原因记 flat", stats.get("why_flat"), 1)

# 重复：第二桶跟已选帧差 DEDUP_BITS 位 -> 换；差 DEDUP_BITS+1 位 -> 留
FRAMES.clear(); AVAILABLE.clear()
frame(1.0, h=0); frame(2.0, h=NEAR); frame(3.0, h=FAR)
frames, stats = M._pick_frames(asset, [[1.0], [2.0, 3.0]], ".", 10.0, (341, 192), None)
check("近似帧被换掉", [f["seconds"] for f in frames], [1.0, 3.0])
check("换人原因记 dup", stats.get("why_dup"), 1)

FRAMES.clear(); AVAILABLE.clear()
frame(1.0, h=0); frame(2.0, h=FAR)
frames, stats = M._pick_frames(asset, [[1.0], [2.0]], ".", 10.0, (341, 192), None)
check("界限外不误杀", [f["seconds"] for f in frames], [1.0, 2.0])
check("界限外不记换人", stats.get("moved"), None)

# 桶内全灭：退回首选，宁可重复也不空格子
FRAMES.clear(); AVAILABLE.clear()
frame(7.0, mean=1.0, h=FAR)
frames, stats = M._pick_frames(asset, [[7.0]], ".", 10.0, (341, 192), None)
check("全灭时退回首选", [f["seconds"] for f in frames], [7.0])
check("全灭时记 kept_worst", stats.get("kept_worst"), 1)

# 抽不出来的时间点跳过，不占格子
FRAMES.clear(); AVAILABLE.clear()
frame(1.0, h=0)
frames, stats = M._pick_frames(asset, [[1.0], [2.0, 3.0]], ".", 10.0, (341, 192), None)
check("抽不到就跳过这格", [f["seconds"] for f in frames], [1.0])
check("抽不到记 miss", stats.get("moved"), None)

print("pick_frames: %d 项检查，%d 项失败" % (ncheck, len(fails)))
for f in fails:
    print(" -", f)
sys.exit(1 if fails else 0)
