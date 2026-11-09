import json
"""取窗与参数校验的回归测试。

这一组全是纯逻辑（不碰模型、不碰素材），跑得很快。
每个用例对应一个实际踩过的 bug，注释里写了当时的现象。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcp_server as M

ASSET = {"duration": 276.0, "width": 1280, "height": 720}
fails = []


def check(name, got, want):
    if got != want:
        fails.append("%s\n    期望 %r\n    实际 %r" % (name, want, got))
        print("  FAIL %s" % name)
    else:
        print("  ok   %s" % name)


# --- seconds <= 0 ---
# 以前 video/subtitles 不查，seconds=0 会静默退化成 1 帧 / 0 段，
# 0 段还被解释成「字幕可能在上方」，让人白试半天 band。
check("seconds=0 被拒", M._pos_seconds({"seconds": 0}) is not None, True)
check("seconds=-5 被拒", M._pos_seconds({"seconds": -5}) is not None, True)
check("seconds 缺省放行", M._pos_seconds({}), None)
check("seconds 非数字被拒", M._pos_seconds({"seconds": "abc"}) is not None, True)

# --- time 越界 ---
# 以前会同时给出「已到视频结尾」和「回退到 0」，自相矛盾。
s, e, note = M._window({"time": 99999}, ASSET)
check("time 越界回到 0", (s, e), (0.0, 276.0))
check("time 越界只有一条 note", note.count("；"), 0)
check("time 越界 note 提回退", "backtracked to 0" in note, True)
check("time 越界不提结尾", "reached the end" in note, False)

# --- time 负数 ---
s, e, note = M._window({"time": -10}, ASSET)
check("负 time 从 0 开始", s, 0.0)
check("负 time 有提示", "negative" in note, True)

# --- 正常窗口 ---
s, e, note = M._window({"time": 10, "seconds": 5}, ASSET)
check("正常窗口", (s, e, note), (10.0, 15.0, ""))

# --- 超过上限 ---
# 注意用 2000 秒的长片测：276 秒的片子会先夹上限再夹结尾，
# 断言 900/1000 会失败，看不出究竟是哪一步生效。
LONG = {"duration": 2000.0, "width": 1280, "height": 720}
s, e, note = M._window({"time": 0, "seconds": 5000}, LONG)
check("超出上限被夹", e, 1000.0)
check("上限有提示", "capped at" in note, True)
# 短片：600 秒请求落在 276 秒里，应该只夹结尾，不提上限
s, e, note = M._window({"time": 0, "seconds": 600}, ASSET)
check("短片夹结尾", e, 276.0)
check("短片不提上限", "上限" in note, False)

# 不传 seconds 时按 DEFAULT_ANALYZE_SECONDS 取窗，而不是整片或上限
s, e, note = M._window({}, LONG, default_span=M.DEFAULT_ANALYZE_SECONDS)
check("默认窗口 100 秒", (s, e, note), (0.0, 100.0, ""))

# --- crop 越界 ---
# 以前 ffmpeg 不报错，只吐一张 295 字节的黑图，还标 ok=True。
box = M._crop_box("100x100+99999+0")
check("_crop_box 解析", box, (100, 100, 99999, 0))
check("_crop_box 非法返回 None", M._crop_box("abc"), None)
try:
    M._parse_crop("0x0+0+0")
    fails.append("crop 零尺寸没报错")
except ValueError:
    print("  ok   crop 零尺寸被拒")

# --- JSON-RPC ---
check("未知工具用 -32602", M.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                    "params": {"name": "nope"}})["error"]["code"], -32602)
check("未知方法用 -32601", M.handle({"jsonrpc": "2.0", "id": 1, "method": "nope"})["error"]["code"], -32601)
check("非法请求用 -32600", M.handle(["x"])["error"]["code"], -32600)
check("parse error 用 -32700", M._parse_error("x")["error"]["code"], -32700)

# --- 时间格式：只到秒 ---
# 毫秒对定位没用，每段却多 8 个字符。98 段字幕就是 784 字符。
check("_tc 到秒", M._tc(83.47), "01:23")
check("_tc 四舍五入", M._tc(59.6), "01:00")
check("_tc 零", M._tc(0), "00:00")
check("_tc 不含毫秒", "." in M._tc(12.345), False)
check("_tc 负数归零", M._tc(-5), "00:00")

# --- SenseVoice 富标签解析 ---
import engine
check("标签剥壳", engine._tag("<|NEUTRAL|>"), "NEUTRAL")
check("标签空值", engine._tag(None), "")
check("标签 NONE", engine._tag("<|NONE|>"), "")
check("标签普通字符串", engine._tag("zh"), "zh")

# --- 面向模型的文字统一用英文 ---
# note / hint / error 都是给模型读的，schema 也是英文，混着来没道理。
import re as _re
_HAN = _re.compile(r"[\u4e00-\u9fff]")
for _args, _lab in (({"time": -5}, "负 time"), ({"time": 99999}, "越界"),
                    ({"time": 270, "seconds": 10}, "报结尾"),
                    ({"time": 0, "seconds": 5000}, "报上限")):
    _n = M._window(_args, ASSET)[2]
    check("%s note 无中文" % _lab, _HAN.search(_n), None)

import inspect as _inspect
_src_audio = _inspect.getsource(M.tool_audio)
check("audio 缺 view 不再报错", 'view is required' in _src_audio, False)
check("缺 view 时落到 level", 'raw_view or "level"' in _src_audio, True)
check("audio 不再有 all 分支", '== "all"' in _src_audio, False)
_audio_schema = [t for t in M.TOOLS if t["name"] == "audio"][0]["inputSchema"]
check("audio schema 不把 view 设为必填", _audio_schema.get("required"), None)
check("audio schema 给 view 写了默认值", _audio_schema["properties"]["view"].get("default"), "level")
check("audio enum 没有 all", "all" in _audio_schema["properties"]["view"]["enum"], False)

# --- 基频：归一化自相关 ---
# 参考图 302.46Hz -> 周期 3.31ms / D#4 / -48.92 音分。
import numpy as _np
_sr = 44100
_t = _np.arange(4096) / float(_sr)
_p = engine._f0_autocorr(_np.sin(2 * _np.pi * 302.46 * _t), _sr)
check("基频 302.46Hz 测得准", abs(_p["freq_hz"] - 302.46) < 0.2, True)
check("周期 3.306ms", abs(_p["period_ms"] - 3.306) < 0.01, True)
check("音名 D#4", _p["note"], "D#4")
check("音分 -48.9", abs(_p["cents"] + 48.92) < 1.0, True)
check("纯音置信度接近 1", _p["confidence"] > 0.95, True)

# 1975Hz 的第一个峰正好落在搜索区间的下端点。只在 [lo, hi] 里找局部极大
# 会漏掉它、退到第二个周期，报成 987Hz（低八度）。
_hi = engine._f0_autocorr(_np.sin(2 * _np.pi * 1975.0 * _t), _sr)
check("接近上限的 1975Hz 不报低八度", abs(_hi["freq_hz"] - 1975.0) < 2.0, True)

# 上限 5000Hz（对齐 phyphox），4500Hz 仍要测准
_p2 = engine._f0_autocorr(_np.sin(2 * _np.pi * 4500.0 * _t), _sr)
check("4500Hz 测得准", abs(_p2["freq_hz"] - 4500.0) < 5.0, True)
check("4500Hz 置信度高", _p2["confidence"] > 0.9, True)

# 谐波丰富时最高峰未必是基频，取「最早够高的峰」才不会被 2 倍频骗走
_saw = sum(_np.sin(2 * _np.pi * 110 * k * _t) / k for k in range(1, 9))
check("锯波仍报 110Hz 基频", abs(engine._f0_autocorr(_saw, _sr)["freq_hz"] - 110.0) < 0.5, True)

check("静音无基频", engine._f0_autocorr(_np.zeros(4096), _sr), None)
check("太短的帧不硬测", engine._f0_autocorr(_np.ones(32), _sr), None)
check("view 支持 pitch", "pitch" in _audio_schema["properties"]["view"]["enum"], True)
check("view 支持 history", "history" in _audio_schema["properties"]["view"]["enum"], True)
check("history 是窗口视图", '"history"' not in '("spectrum", "wave", "pitch")', True)

# --- 数峰法（phyphox 口径）---
# 周期 = 最后一个峰位置 / 峰数；100ms 窗在 4410 点数下比 4096 点更准。
_t100 = _np.arange(int(_sr * 0.1)) / float(_sr)
_pb = engine._f0_autocorr(_np.sin(2 * _np.pi * 1000.0 * _t100), _sr)
check("1000Hz 报 1000", abs(_pb["freq_hz"] - 1000.0) < 0.1, True)
check("1000Hz 峰数约 20", abs(_pb["peaks"] - 20) <= 1, True)
check("峰数已上报", "peaks" in _pb, True)
_pc = engine._f0_autocorr(_np.sin(2 * _np.pi * 4500.0 * _t100), _sr)
check("4500Hz 在 100ms 窗也准", abs(_pc["freq_hz"] - 4500.0) < 2.0, True)
check("无周期门槛是 0.25", engine.F0_MIN_CONF, 0.25)
_pt = engine.pitch_track(0.5 * _np.sin(2 * _np.pi * 440.0 * _np.arange(_sr) / float(_sr)), _sr, 0.0)
check("Frequency History 出轨迹", len(_pt) > 5, True)
check("轨迹是 (t, f, conf)", (len(_pt[0]), abs(_pt[0][1] - 440.0) < 1.0), (3, True))
check("pitch 也走单帧路径", '("spectrum", "wave", "pitch")' in _inspect.getsource(M.tool_audio), True)
# 参数校验在取素材之前：报错要指向参数，别先喊「open a video first」
_bad = json.loads(M.tool_audio({"view": "bogus"})[0])
check("未知 view 直接报参数错", "unknown view" in _bad.get("error", ""), True)
check("报错里列全了 view", "| pitch" in _bad.get("error", ""), True)

# --- spectrum/wave/pitch 只认时间戳 ---
# 这三个视图读一个 100ms 帧（FRAME_MS，phyphox 口径），seconds 对它没有意义：
# 传了只会在更大范围里挑最响的一帧，报出来的值却像整段窗口的代表值。
_src_audio2 = _inspect.getsource(M.tool_audio)
check("单帧长度走 FRAME_MS 常量", "engine.FRAME_MS" in _src_audio2, True)
check("单帧不再写死 0.093", "0.093" in _src_audio2, False)
check("传了 seconds 会说明被忽略", 'seconds is ignored' in _src_audio2, True)
check("帧长 100ms", engine.FRAME_MS, 100.0)
check("Frequency History 默认 20ms", engine.HISTORY_MS, 20.0)

# --- frame_ms 对所有视图都生效 ---
_src3 = _inspect.getsource(M.tool_audio)
check("frame_ms 在 schema 里", "frame_ms" in _audio_schema["properties"], True)
check("frame_ms 是 5-200", "5-200" in _audio_schema["properties"]["frame_ms"]["description"], True)
check("level 块长独立于 FFT", engine.LEVEL_MS, 20.0)
check("analyze_spectrum 收 level_ms", "level_ms" in _inspect.signature(engine.analyze_spectrum).parameters, True)
check("analyze_spectrum 收 fft_ms", "fft_ms" in _inspect.signature(engine.analyze_spectrum).parameters, True)
check("analyze_spectrum 收 fmin", "fmin" in _inspect.signature(engine.analyze_spectrum).parameters, True)
# 帧长 -> FFT 点数：25ms @ 48k = 1200
_sr48 = 48000
_x = 0.5 * _np.sin(2 * _np.pi * 440.0 * _np.arange(_sr48) / float(_sr48))
_c = engine._channel_metrics(_x, _sr48, 0.0, 1200, 5.0)
check("level 5ms 一帧 200 点", _c["level"]["times"][1] - _c["level"]["times"][0], 0.005)
_c2 = engine._channel_metrics(_x, _sr48, 0.0, 1200, 100.0)
check("level 100ms 一帧", round(_c2["level"]["times"][1] - _c2["level"]["times"][0], 3), 0.1)
check("FFT 块长仍是 25ms", _c["wave"]["ms"], 25.0)
check("默认 fft_size 按采样率折算", engine.analyze_spectrum.__defaults__[0], None)

# --- 音频图尺寸 ---
# 上下两半必须正好拼成正方形，且 blit 偏移跟着 H//2 走（别留硬编码 512）。
import audiochart as _ac
check("音频图边长 768", _ac.FIG_W, 768)
_src_render = _inspect.getsource(_ac.render)
check("下半接 H//2 而非写死 512", "c.blit(bot.buf, 0, H // 2)" in _src_render, True)
check("分隔线也在 H//2", "c.hline(H // 2" in _src_render, True)
check("render 不留硬编码 1024", "1024" in _src_render, False)

# --- 音频轨迹抽稀 ---
# 原来是 step = max(1, len//60)：len<120 时整除恒为 1，等于没抽稀，
# 8 秒的片子就给 62 个点、15 秒能到 81 个。改成 ceil + 目标 40 点。
import math as _math
for _n in (31, 62, 100, 161, 300):
    _pts = len(range(0, _n, max(1, _math.ceil(_n / 40.0))))
    check("轨迹点数有上限 (len=%d → %d)" % (_n, _pts), _pts <= 41, True)
check("轨迹改用 ceil 抽稀", "math.ceil(len(track) / 40.0)" in _inspect.getsource(M.tool_audio), True)

# 工具返回值一律不缩进：缩进和 ", " / ": " 里的空格不提供信息，只吃 token。
# indent=None 不够，它只管换行，分隔符旁的空格还在，所以要看 separators。
_dense = M._dump({"a": [1, 2], "b": [3, 4]})
check("返回没有换行", "\n" in _dense, False)
check("返回没有分隔符空格", " " in _dense, False)
check("_dump 不再有 compact 开关", "compact" in _inspect.signature(M._dump).parameters, False)
check("工具里没有 indent=2 的返回", "indent=2" in _inspect.getsource(M), False)

# 每个 view 只发自己那几块数：level 的返回里不该再出现 pitch / peak_freq_track
_src_audio = _inspect.getsource(M.tool_audio)
check("按 view 裁剪返回", "_PER_VIEW.get(view, ())" in _src_audio, True)
check("level 只带 level", '"level": ("silent", "level")' in _src_audio, True)
check("spectrum 只带频谱", '"spectrum": ("spectrum",)' in _src_audio, True)
check("wave 只带波形统计", '"wave": ("wave",)' in _src_audio, True)
check("基频只给 pitch 视图", 'if view == "pitch":' in _src_audio, True)
check("历史只给 history 视图", 'elif view == "history":' in _src_audio, True)

# --- 单帧取窗：别为不存在的窗口报警 ---
# 以前 quantity=1 时仍走完整窗口逻辑，t=100 会附赠一句「已到视频结尾 276 秒」，
# 但单帧只用起点，结尾根本不参与取图。
s, e, note = M._window({"time": 100}, ASSET, single=True)
check("单帧不报结尾", note, "")
check("单帧起点终点重合", (s, e), (100.0, 100.0))
check("单帧忽略 seconds 上限", M._window({"time": 100, "seconds": 5000}, ASSET, single=True)[2], "")
check("单帧仍报负时间", "negative" in M._window({"time": -5}, ASSET, single=True)[2], True)
check("单帧仍报越界回退", "backtracked to 0" in M._window({"time": 99999}, ASSET, single=True)[2], True)
# 多帧时该报的还得报
check("多帧报结尾", "reached the end" in M._window({"time": 270, "seconds": 10}, ASSET)[2], True)

# --- 声学字段的边界 ---
# emotion / event / lang 来自 SenseVoice，只有 stt 模式有。
# OCR 读像素，原理上取不到——这条边界要是破了，就是编造数据。
import inspect
src_ocr = inspect.getsource(M._subs_ocr)
src_stt = inspect.getsource(M._subs_stt)
check("OCR 不写 emotion", "emotion" in src_ocr, False)
check("OCR 不写 event", '"event"' in src_ocr, False)
# 字段语义写在 schema 里（只付一次），不进每次响应——调用方自己知道传的哪个 mode。
check("OCR 不重复声明字段缺失", "acoustic_fields" in src_ocr, False)
check("STT 不重复声明字段存在", "acoustic_fields" in src_stt, False)
check("STT 确实带 emotion", '"emotion"' in src_stt, True)
# note 只能有一处赋值。之前 window_note 那条在后面无条件覆盖，BGM 提示被吞掉了。
check("STT 的 note 只赋值一次", src_stt.count('out["note"] ='), 1)

# --- 响应里不许回显已知事实 ---
# schema 是「调用前读一次」，响应是「每次调用都读」。
# 已知不变的东西进前者；把已知事实塞回响应，等于每次付费买一遍废话。
for bad, why in [
    ('"engine"', "stt 只有一种引擎，属 schema"),
    ('"range"', "就是调用方传的 time/seconds；被夹了 note 会说"),
    ('"text": "".join', "等于把 segments 再拼一遍"),
    ('band_auto', "恒等于 band，零额外信息"),
]:
    check("STT 不回显 %s" % bad, bad in src_stt, False)
# 查 JSON 键而不是裸词——注释里提到字段名是合法的，写进输出才是问题
check("OCR 不输出 band_auto 键", '"band_auto":' in src_ocr, False)

# --- 竖屏的分辨率判断 ---
# 竖屏 height 是长边。用 [height<=720] 过滤会把竖屏全杀掉、静默回落到全尺寸；
# 改用 min(w,h) 又根本没有这个语法（SyntaxError）。现在走 format_sort res:，
# 而 yt-dlp 的 res 本来就是 min(height, width)。
src_open_full = inspect.getsource(M.tool_open)
check("下载不用 height 过滤（会误杀竖屏）", "[height<=" not in src_open_full, True)
check("下载不用不存在的 min()", "min(w,h)" not in src_open_full, True)
check("下载用 res: 排序短边", '"res:%d"' in src_open_full, True)
check("不写死格式过滤（避免无匹配回落）", 'fmt = "bv*+ba/b"' in src_open_full, True)
check("quality 档位仍是短边语义", M._QUALITY_MAP("medium"), 720)
# detail 是图片质量（像素预算），和下载分辨率的 quality 是两回事
check("video detail 默认 768", M.VIDEO_DETAIL_PX[M._detail_level({}, M.VIDEO_DETAIL_PX, "low")], 768)
check("detail 不再认旧名 quality", M._detail_level({"quality": "high"}, M.VIDEO_DETAIL_PX, "low"), "low")

# --- schema 的必填要和实际行为一致 ---
# source 只在 action=open 时才必要：写成无条件 required，list 就得凭空编路径；
# 完全不写，模型只能靠试错。用 allOf 条件必填把「缺省 action 也是 open」表达出来。
_w = [t for t in M.TOOLS if t["name"] == "workspace"][0]["inputSchema"]
check("workspace 有条件必填", "allOf" in _w, True)
check("open 时要求 source", _w["allOf"][0]["then"]["required"], ["source"])
check("条件挂在 action=open 上", _w["allOf"][0]["if"]["properties"]["action"]["const"], "open")
check("action 本身不是无条件必填", "required" in _w, False)
# 运行时兜底也要说得清楚，别让模型猜
_missing_src, _ = M.tool_open({})
check("缺 source 报错点明原因", "source is required" in _missing_src, True)
_a = [t for t in M.TOOLS if t["name"] == "audio"][0]["inputSchema"]
# view 缺省给 level：最便宜（27KB/0.8s）、按窗口算（认 seconds）、没有歧义。
# spectrogram 384KB 当默认等于替调用方做了个很贵的决定；
# spectrum/wave/pitch 只读一帧，会把 seconds 悄悄丢掉。
# history 也按窗口算（别把它归进单帧那一拨），只是逐帧追基频更贵、更假设素材带音高。
check("audio 不再硬必填 view", _a.get("required"), None)
check("view 的缺省是 level", _a["properties"]["view"].get("default"), "level")
_src_audio = inspect.getsource(M.tool_audio)
# 默认值会被无声接受，所以走了默认就在 note 里说一句
check("走缺省时 note 里说清楚", 'view defaulted to %s' in _src_audio, True)
check("显式传 view 时不加 note", "if view_defaulted:" in _src_audio, True)
# history 是窗口视图，不是单帧：它走 _window(default_span=15.0, cap=15.0)，
# seconds=2 和 seconds=8 应该拿到不同的 range。曾经在注释里把它写成"只读一帧"，是错的。
check("单帧分支只有 spectrum/wave/pitch", '("spectrum", "wave", "pitch")' in _src_audio, True)
check("history 不在单帧分支", '("spectrum", "wave", "pitch", "history")' in _src_audio, False)
check("history 按窗口取", 'default_span=15.0, cap=15.0' in _src_audio, True)
check("detail 认新名", M._detail_level({"detail": "max"}, M.VIDEO_DETAIL_PX, "low"), "max")
check("detail 乱填回落默认", M._detail_level({"detail": "huge"}, M.VIDEO_DETAIL_PX, "low"), "low")
check("音频 detail 四档", sorted(M.AUDIO_DETAIL_PX.values()), [768, 1024, 1280, 1536])
check("音频默认 low=768（和以前一样大）", M.AUDIO_DETAIL_PX[M._detail_level({}, M.AUDIO_DETAIL_PX, "low")], 768)

# --- 格子尺寸：整页预算要按整页算 ---
# 以前把 sqrt(cap/(aspect·cols·rows)) 当"短边"，公式已除过 cols·rows，
# 得到的其实是单格窄边。竖屏再拿它当页面宽 → 整页 316% 预算。
# 横屏恰好 100%，所以只有竖屏能暴露。
cap = M._max_pixels({"detail": "medium"})
for label, asp, c, r in (("竖屏3x2", 1080/1920, 3, 2), ("竖屏2x5", 1080/1920, 2, 5),
                         ("横屏3x3", 1280/720, 3, 3), ("横屏5x2", 1280/720, 5, 2),
                         ("方形3x3", 1.0, 3, 3)):
    w, h = M._sheet_cells({"detail": "medium"}, asp, c, r)
    got = w * h * c * r
    check("%s 整页 ≈ 预算" % label, abs(got / cap - 1.0) < 0.06, True)
    check("%s 单格保持画面比例" % label, abs((w / h) / asp - 1.0) < 0.02, True)

# --- 工作区清理 ---
# _prune_workspaces 曾经定义了但零调用，MVV_KEEP_WORKSPACES 从来没生效，
# 目录只增不减。这里既测排序，也测它确实被接进了打开流程。
import tempfile, json as _json, os as _os, time as _time
with tempfile.TemporaryDirectory() as _td:
    for i in range(1, 8):
        d = _os.path.join(_td, "ws%d" % i)
        _os.makedirs(d)
        with open(_os.path.join(d, "meta.json"), "w") as f:
            _json.dump({"id": "ws%d" % i,
                        "opened_at": _time.strftime("2026-11-0%dT10:00:00" % i)}, f)
    gone = M._prune_workspaces(_td, keep=5)
    left = sorted(_os.listdir(_td))
    check("清理删掉最旧的 2 个", sorted(gone), ["ws1", "ws2"])
    check("清理后只剩 5 个", len(left), 5)
    check("清理保留最新的", left, ["ws3", "ws4", "ws5", "ws6", "ws7"])
    check("keep 大于总数时不删", M._prune_workspaces(_td, keep=99), [])

# opened_at 优先于文件 mtime：meta 被别的操作重写过，mtime 就不是打开时间了
with tempfile.TemporaryDirectory() as _td:
    for name, at in (("old", "2026-01-01T00:00:00"), ("new", "2026-06-01T00:00:00")):
        d = _os.path.join(_td, name)
        _os.makedirs(d)
        with open(_os.path.join(d, "meta.json"), "w") as f:
            _json.dump({"id": name, "opened_at": at}, f)
    _os.utime(_os.path.join(_td, "old", "meta.json"), None)  # 旧目录反而 mtime 更新
    check("按 opened_at 排序而非 mtime", M._prune_workspaces(_td, keep=1), ["old"])

# 清理必须接在打开成功之后，不能只是定义了没人调
src_open = inspect.getsource(M.tool_open) + inspect.getsource(M._opened)
check("打开流程会触发清理", "_prune_workspaces" in src_open, True)

# --- 按画面变化取帧 ---
# 策略：分数从高到低取，够 N 就停；不够则按变化点间隔插值补齐并告知。
sparse = [(i * 0.5, 0.001) for i in range(21)]
sparse[10] = (5.0, 0.9)
sparse[20] = (10.0, 0.8)
ts, filled = M._pick_scene(sparse, 9, 0, 10)
check("稀疏场景补齐数", filled, 7)
check("稀疏场景凑满 9 格", len(ts), 9)
check("稀疏场景保留变化点", (5.0 in ts) and (10.0 in ts), True)
check("稀疏场景时间有序", ts, sorted(ts))

# 低分噪声不该占位：它们离已选点远，不用相对下限的话会永远凑满，
# 补齐分支永远不触发、警告永远不出现。
noisy = [(i * 0.1, 0.0001) for i in range(100)]
noisy[30] = (3.0, 0.9)
ts, filled = M._pick_scene(noisy, 5, 0, 10)
check("噪声不冒充满额", filled > 0, True)

dense = [(i * 0.5, 0.1 + (i % 7) * 0.1) for i in range(21)]
ts, filled = M._pick_scene(dense, 9, 0, 10)
check("变化充足时不硬补", filled < 9, True)

# 同一场景连闪 20 帧：靠 min_gap 压成 1 个，不该占满 5 格
flash = [(i * 0.02, 0.95 if 100 <= i < 120 else 0.01) for i in range(300)]
ts, filled = M._pick_scene(flash, 5, 0, 6)
check("同场景连闪只算一处变化", filled, 4)

# 补点必须落在窗口内且不重复
ts = M._fill([3.0, 7.0], 6, 0.0, 10.0)
check("补点数对", len(ts), 6)
check("补点不越界", all(0.0 <= t <= 10.0 for t in ts), True)
check("补点不重复", len(set(ts)), len(ts))
check("补点保留锚点", (3.0 in ts) and (7.0 in ts), True)
check("n=1 不走场景分析", M._pick_scene([(0, 0.9)], 1, 2.0, 8.0), ([2.0], 0))

# --- 输出里不许有绝对路径 ---
# 图片走 image 内容块，模型读不了本地路径。绝对路径前缀（workspace/<标题>/video/）
# 每处 55 字符，quantity=9 时 12 处 = 1333 字符，占文本 63%。
src_video = inspect.getsource(M.tool_video)
check("video 不输出绝对路径", '"save": f["path"]' in src_video, False)
check("video 不输出 videoDir", '"videoDir"' in src_video, False)
check("video 的 timeline 只放时间", '"path": f["path"]' in src_video, False)
# 输出只留模型用得上的。这些字段要么恒为 true、要么是入参回显、
# 要么能数出来、要么它根本没有文件系统去用。
for dead in ('"ok": True', '"mode": "', '"title": asset.get', '"frames": len(',
             '"grid": [cols', '"cellSize"', '"crop": spec', '"detail": a.get',
             '"sheetKB"', '"file": Path('):
    check("video 不输出 %s" % dead.strip(), dead in src_video, False)
check("video 保留 timeline", '"timeline"' in src_video, True)
check("video 保留 note", '"note"' in src_video, True)

# workspace list：绝对路径模型读不了，bytes 它也做不了决定
src_status = inspect.getsource(engine.status)
check("list 不输出 workspace 绝对路径", '"workspace": str(root)' in src_status, False)
check("list 不输出每项 path", '"path": m.get' in src_status, False)
check("list 不输出 bytes", '"bytes":' in src_status, False)

# audio：mono 时 right 与 left 逐字节相同，不该发两份
src_audio = inspect.getsource(M.tool_audio)
check("audio 单声道不重复发 right", 'if not d["mono"]:' in src_audio, True)
check("audio 不输出图片文件名", 'summary["images"]' in src_audio, False)
check("audio 不输出恒真的 ok", '"ok": True' in src_audio, False)

# subtitles：segment_count 就是 len(segments)
src_subs = inspect.getsource(M.tool_subtitles)
check("subtitles 不重复给 segment_count", 'segment_count' in src_subs, False)

# 缓存键必须含实际取到的帧时间点：按画面变化取帧时，同一窗口两次调用
# 可能取到完全不同的帧，键里不写 ts 就会拿旧图冒充新结果。
check("拼图缓存键含帧时间", "ts_key" in src_video, True)
check("pick=even 走等分", "even" in inspect.getsource(M._scene_times), True)

print("\nRESULT", "ALL OK" if not fails else "HAS FAIL (%d)" % len(fails))
for f in fails:
    print(" -", f)
sys.exit(0 if not fails else 1)

# --- history 帧长：默认 20ms，也可指定 ---
_t1s = _np.arange(_sr) / float(_sr)
_q = 0.5 * _np.sin(2 * _np.pi * 300 * _t1s)
_t10 = engine.pitch_track(_q, _sr, 0, frame_ms=10, step_ms=10)
_t25 = engine.pitch_track(_q, _sr, 0, frame_ms=25, step_ms=25)
_t20 = engine.pitch_track(_q, _sr, 0, frame_ms=20, step_ms=20)
check("frame_ms=10 步进 10ms", round(_t10[1][0] - _t10[0][0], 3), 0.01)
check("frame_ms=25 步进 25ms", round(_t25[1][0] - _t25[0][0], 3), 0.025)
check("默认 20ms 步进 20ms", round(engine.pitch_track(_q, _sr, 0)[1][0], 3), 0.02)
check("25ms 帧点数约为 10ms 的四成", len(_t25) <= 45 and len(_t10) >= 95, True)
# 25ms 能装下 50Hz 的整周期（20ms），10ms 装不下
_low = 0.5 * _np.sin(2 * _np.pi * 50 * _t1s)
check("50Hz 在 20ms 帧下可测", abs(engine.pitch_track(_low, _sr, 0, frame_ms=20)[0][1] - 50) < 0.5, True)
check("50Hz 在 25ms 帧下可测", abs(engine.pitch_track(_low, _sr, 0, frame_ms=25)[0][1] - 50) < 0.5, True)
check("50Hz 在 10ms 帧下测不到", engine.pitch_track(_low, _sr, 0, frame_ms=10), [])
