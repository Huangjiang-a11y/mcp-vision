# mcp-vision

[![CI](https://github.com/Huangjiang-a11y/mcp-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/Huangjiang-a11y/mcp-vision/actions/workflows/ci.yml)

**让 AI 看视频。** 纯 Python 的 MCP 服务器，不依赖 Node。

给一个本地文件或 B 站链接，它把片子拆成模型能直接吃的东西：带时间戳的帧、一页网格
故事板、音频电平与频谱、画面 OCR、语音转写。全部在本机跑——ffmpeg 解码、numpy 计算、
onnxruntime 跑 PP-OCRv4 / SenseVoice，没有浏览器、没有 Node 进程、没有 Electron。

> 仅供学习研究使用，MIT。下载内容的版权归原作者。

## 依赖

| 必需 | 说明 |
|---|---|
| Python 3.10+、ffmpeg / ffprobe | 解码与探测 |
| numpy | 音频与图像计算 |
| onnxruntime | OCR（`pkg install python-onnxruntime`） |

可选：`yt-dlp`（链接下载）、`sherpa_onnx`（语音转写）。模型不进仓库，路径用环境变量给
（`MVV_OCR_*` / `MVV_ASR_*`），详见 [INSTALL.md](INSTALL.md)。

```sh
sh check.sh              # 环境体检
python3 mcp_server.py    # stdio
python3 mcp_server.py --http 8765   # HTTP，默认只绑 127.0.0.1
```

## 工具

四个工具，`workspace` 开头，其余三个作用于最近打开的那支片子（可用 `video_id` 指定）。

| 工具 | 做什么 |
|---|---|
| `workspace` | `open`（URL 下载 / 本地路径，`quality` 选下载高度、`cookies` 用 cookies.txt）/ `list` |
| `video` | 一页 `quantity` 格（默认 9，上限 36）；`quantity=1` 给单帧原图，可 `crop` |
| `audio` | 一张图 + 图背后的数字，`view` 选测量方式，缺省 `level` |
| `subtitles` | `mode=ocr` 读画面上的字（`band` 可指定字幕带，'auto' 或比例）/ `mode=stt` 转写语音 |

**`video` 默认按时间分桶取帧**：窗口均分 n 格，每格按画面变化分数挑代表帧（变化点
稀缺的格子取格中点）；太暗 / 纯色、或与已选帧几乎相同的换同格备选，换不到留首选。
换了几格、为什么，写进 `note`。`pick=even` 退回等分。

**`detail` 是图片质量，`quality` 是下载分辨率**，两者无关：前者是等面积正方形边长
（low 768 / medium 1024 / high 1280 / max 1536，只缩不放），后者按短边筛（竖屏不误杀）。

**`audio` 的 `view`**：`level`（20ms RMS 电平）/ `spectrum`（一帧的幅度谱）/
`spectrogram`（时频）/ `peak`（峰值频率轨迹）/ `wave`（原始波形）/
`pitch`（一帧的基频：周期 / 音名 / 音分 / 置信度）/ `history`（基频随时间，20ms 块，只看趋势）。
`spectrum`、`wave`、`pitch` 只读 `time` 处那一帧，`seconds` 被忽略并在 `note` 里说。
缺省给 `level`：按窗口算、27KB、0.8s，最便宜也最没歧义（`spectrogram` 一张 384KB）。

## 取窗

| 参数 | 含义 |
|---|---|
| `time` | 从哪开始，默认 0 |
| `seconds` | 看多久，默认 100，上限 1000 |

超限会被夹住并在 `note` 里说夹到哪。长片靠 `time=0, 1000, 2000…` 翻页。
一页的代价是「页」不是「秒」：通读时长片按上限翻页，只看一段就收窄窗口再提 `quantity`。

## 输出

- **不裁剪**：`_dump()` 默认截断 0 字（旧的 8000 字符上限会把 97 段字幕砍成 55 段还不显眼）。
- **紧凑 JSON**：无缩进、分隔符不带空格。读者是模型，缩进只吃 token。
- **时间只到秒**：`{"from": "MM:SS", "to": "MM:SS", "text": ...}`，OCR 与转写同形。
  毫秒对定位没用，98 段就是白花 784 字符。
- 返回是 `content` 数组：先一段紧凑 JSON，图片跟在后面（帧 `image/jpeg`、音频图 `image/png`）。
- 产物落在 `workspace/<标题>/{video,audio,subtitles}/`，可以直接 `ls` / `rm`。

`mode=ocr` 与 `mode=stt` 拿到的东西不一样，这不是没实现，是原理上取不到：

| | ocr | stt |
|---|---|---|
| 读的是 | 画面像素 | 声音波形 |
| 附带的 | `conf` | `emotion` / `event`（BGM 也算事件）/ `lang` / `summary` |

## 音频图的读数口径

- FFT 按 100ms 折算（44.1k → 4410 点，分辨率 10.0 Hz）；`frame_ms` 可改 5–200。
- 幅度满幅正弦 = 1，不按窗口峰值归一；纵轴顶端 = 峰值向上取整到 10 的整数次幂。
- 峰值频率取 `time` 处那一帧，对数域抛物线插值，并补汉宁窗的扇贝损失。
- 基频**按声道各测**，不混单声道：L 266.9Hz C4 + R 375.0Hz F#4 混起来会报 320.9Hz，
  两边都没有这个音。峰用「数峰法」（最大的一半以上的局部极大，最后一个 ÷ 个数），
  不加窗（55Hz 上加汉宁会偏 21 音分）。
- 已知偏差：时频图噪声底比读数低 1.76 dB（纯音标定 + ENBW 1.5 的必然），不动。
- `pitch` 的方法参考 phyphox 的三篇实验文档，代码独立实现；本工具多做了能量归一化，
  所以 `curve` 纵轴不等同于 phyphox 界面的相对值。

## 性能

实测 Android aarch64（8 核）/ Python 3.14 / onnxruntime 1.31.1。

**线程**：视频 / 音频工具 3 条解码（`MVV_FFMPEG_THREADS`），字幕 2 条 × 2 进程
（`MVV_OCR_FFMPEG_THREADS`），ORT 2 条（`MVV_ORT_THREADS`，det 推理 2 线程 0.117s、
8 线程反而 0.149s）。

| 场景 | 数字 |
|---|---|
| `scene_scores` 0-200s | 2 条 10.80s → **3 条 9.48s** → 8 条 8.69s |
| 字幕 90s 窗口 | 串行 18.97s → **2 进程 15.11s**；与 600s 扫描并发时 32.3 → 23.0s |
| 30s 测试片（15 条字幕） | 全帧 8.10s → **带内跳帧 2.20s** |
| 900s 窗口字幕 | 119s → 95s（双进程），逐字不变 |
| 654s 全片 9 格 | 6 格同镜头 → **2 格**；9 格 3.6→5.3s、36 格 16.0→16.9s |

内存全部按需加载：启动 36MB / 首次 OCR 68MB / 首次出字 90MB / 转写 340MB。
纯静音视频 0.17 秒结束，不加载识别模型。

## 踩过的坑（一行一条，细节看提交）

- **缓存当基线**：第一次跑快一个数量级，是命中了帧图 / `.scene-*.json`。要 A/B 就清干净。
- **`-threads` 管下一个输入**：多输入那条命令得每个 `-i` 前各写一个。
- **量进程池的脚本要有 `__main__` 守卫**：否则子进程重跑脚本，能量出「并行慢一倍」。
- **抬阈值不是分辨**：整页判定抬簇数，结果是「该报的也不报了」；看那条带自己像不像字幕。
- **docstring 里的数字必须测过**：我编过一个「差 20 多位」，实测 0.92 位。
- **别信单帧**：960 宽的预览图里解不出整页的簇，整页判定只能靠多帧聚合。
- **`_prune_workspaces` 曾是死代码**：定义了没人调，`MVV_KEEP_WORKSPACES` 等于假开关。
- **`.mvv-cache` 曾是只进不出**：下载件拷进工作区后还留着，一支 51MB 的片占两份。
- **字幕带检测别给兜底**：0.45 的兜底会把画面里的大字当字幕；检不到就说没有。
- **高度基准用 `max` 不是中位数**：中位数多留下的恰恰是游戏 UI 的边缘文字。
- **彩色像素抹黑再识别**：修好一例红心，整行彩色字幕丢掉分界，错的比对的还多，撤了。
- **识别器升级三连撤**：server_rec_doc 太大、默认切换不可预期、两级复核复杂度不划算。

## 协议与安全

- JSON-RPC 2.0，`protocolVersion: 2024-11-05`，只暴露 `tools`。
- stdio 一行一请求；HTTP 每请求一线程，同名工具排队、不同工具真并行（各一把锁）。
- **无鉴权、无 TLS**：默认绑 `127.0.0.1`。用 `--host 0.0.0.0` 是对外监听，请自己套反代和认证。
- 错误分两层：输入问题回 `{"ok": false, "error": ...}` 文本；协议问题回 JSON-RPC 错误码
  （`-32700` / `-32600` / `-32601` / `-32602` / `-32000`）。
- 子进程全用参数列表，没有 `shell=True`；`crop` 在 Python 侧先与帧尺寸比对，越界直接拒
  （ffmpeg 越界不报错，只吐一张黑图）。
- 工作区内的路径按仓库根 resolve，绝对路径和 `..` 都不拦——这是给本机模型用的工具，不是沙箱。
- 下载缓存只做中转：拷进工作区后删掉，无主的一小时后扫掉。

## 测试

`sh check.sh --test` 跑 6 个模块，不需要模型和素材（CI：ubuntu / Python 3.10 + 3.13）。

| 文件 | 行数 | 覆盖 |
|---|---|---|
| `test_windows.py` | 500 | 取窗与参数校验、错误码、紧凑 JSON、分桶、拼页预算、工作区与缓存清理、schema 必填、频谱与基频精度 |
| `test_page_like.py` | 121 | OCR 框聚类、整页判定与四个反例 |
| `test_stereo.py` | 58 | 合成立体声：左右不串、单声道标注、基频按声道 |
| `test_keep_wide.py` | 53 | 字幕行筛选：窄句不丢、离群不抬标尺 |
| `test_ff_threads.py` | 116 | 7 条 ffmpeg 都带 `-threads`，每个 `-i` 前各一条；两条开关不同名 |
| `test_pick_frames.py` | 135 | 选帧三道筛子：分桶、黑帧、去重 |

没测的：真实素材的端到端、scene 质量的 F1、OCR / STT 精度（要模型）、HTTP 集成。

## 状态

v0.4：四个工具全通。未做 B 站元数据 / 评论 / 弹幕、镜头切分、多站登录态。

思路参考 [video-understanding-kit](https://github.com/yangjili6/video-understanding-kit)、
[dsh-video-vision](https://github.com/Tests4gptplay/dsh-video-vision)、
[dsh-bili-watch](https://github.com/ppppiii/dsh-bili-watch)；代码全部按 Python 重写，运行时不依赖它们。

模型：PP-OCRv4（PaddleOCR，Apache-2.0）、SenseVoice（FunASR，© 阿里巴巴；权重是
FunASR Model License v1.1，须保留模型名与署名）、Silero VAD（MIT）、sherpa-onnx（Apache-2.0）。
