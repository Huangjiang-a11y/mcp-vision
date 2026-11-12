# mcp-vision

[![CI](https://github.com/Huangjiang-a11y/mcp-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/Huangjiang-a11y/mcp-vision/actions/workflows/ci.yml)

**让 AI 看视频。** 纯 Python 的 MCP 服务器，不依赖 Node 运行时。

给它一个本地文件或 B 站链接，它把片子拆成模型能直接吃的东西：带时间戳的帧、
一页网格故事板（默认 9 格）、音频电平与频谱、画面 OCR 与折叠后的字幕轨。
全部在本机完成 —— ffmpeg 解码、numpy 计算、onnxruntime 跑 PP-OCRv4，
没有浏览器、没有 Node 进程、没有 Electron。

四层（`workspace` / `video` / `audio` / `subtitles`）与工作区目录布局参照
[dsh-video-vision](https://github.com/Tests4gptplay/dsh-video-vision) 重写；
结构照搬，代码全部自己写，不 import 任何外部仓库。

> 仅供学习研究使用。代码以 MIT 发布。

## 设计原则

- **纯 Python**：stdlib + numpy + onnxruntime（可选）+ ffmpeg/ffprobe + yt-dlp。无 Node、无 Pillow、无 ImageMagick。
- **自含**：结构照搬，代码重写，不 import 任何外部仓库（见「致谢」）。
- **MCP 优先**：stdio + HTTP 双入口，工具 schema 与实现一一对应。
- **MIT**。

## 依赖

- Python 3.10+（开发在 3.14）
- `ffmpeg` / `ffprobe`（PATH 或配置）
- `numpy`
- `onnxruntime`（可选，OCR 用）
- `sherpa-onnx`（可选，转写用）
- `yt-dlp`（可选，链接下载）

模型不随仓库分发，路径全部走环境变量：

| 变量 | 用途 |
|---|---|
| `MVV_OCR_DET` / `MVV_OCR_REC` / `MVV_OCR_KEYS` | PP-OCRv4 三个文件 |
| `MVV_ASR_MODEL` / `MVV_ASR_TOKENS` / `MVV_VAD_MODEL` | SenseVoice + silero VAD |

没配就只缺对应能力，抽帧和电平分析不受影响。详见 [INSTALL.md](INSTALL.md)。

其余开关（都给了能用的默认值）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MVV_FFMPEG_THREADS` | 3 | 视频 / 音频工具的解码线程 |
| `MVV_OCR_FFMPEG_THREADS` | 2 | 字幕取帧的解码线程（2 进程 × 2 条） |
| `MVV_OCR_WORKERS` | 2 | 字幕 OCR 进程数；起不来就回退串行 |
| `MVV_ORT_THREADS` | 2 | 推理线程 |
| `MVV_KEEP_WORKSPACES` | 5 | 留几个工作区，多的整目录删掉 |
| `MVV_BUCKET_K` | 3 | 每个时间桶留几个候选帧；1 = 关掉换人 |
| `MVV_REC_MAX_W` | 320 | 识别输入最大宽度，手写长行别往下调 |

## 快速开始

```sh
sh check.sh
python3 mcp_server.py            # stdio
python3 mcp_server.py --http 8765
```

`sh check.sh --test` 跑一遍体检 + 不需要模型的回归测试（L/R 分层出图）。
基准脚本在 `scripts/`（需要模型或运行中的服务），回归测试在 `tests/`。

依赖、模型下载、客户端配置见 **[INSTALL.md](INSTALL.md)**；
使用范围与责任见其中的「免责声明」。

## 工具

四个大工具，按职责分层：

- `workspace`：`open`（打开/下载）/ `list`
- `video`：给窗口 + 张数，拼成一页网格。默认**按时间分桶取帧**：窗口均分 n 格，
  每格按画面变化分数挑一个代表帧（变化点稀缺的格子取格中点）；太暗 / 纯色、或与已选帧
  几乎相同的格子会换成同格内的备选帧，换不到就留首选。换了几格、为什么，都写进 `note`。
  `pick=even` 退回按时间等分。
  `quantity=1` 时不出页，直接给单帧原图（可 `crop`）
  - `quantity` 抽帧数（默认 9，上限 36；1 = 单帧）；`detail` 图片质量（像素预算）：low=768（默认）medium=1024 high=1280 max=1536，数值是等面积正方形的边长（拼页时管整页、按格子均分，单帧时管裁剪后那张），只缩不放；下载分辨率**另有一个** `quality`（按短边筛，竖屏不误杀），两者无关，别混
- `audio`：按 `view` 出图，一次一张，**缺省 `level`** —— `level`（20ms RMS 电平）/
  `spectrum`（FFT 幅度谱）/ `spectrogram`（时频图）/ `peak`（峰值频率轨迹）/ `wave`（原始波形）/
  `pitch`（按声道测基频：周期 / 频率 / 音名 / 音分 / 置信度，附自相关曲线；双声道上下两半）/
  `history`（基频 / 周期随时间变化：20 ms 短窗逐帧，只看趋势；`frame_ms` 可改 5–200；按声道各一份，双声道上下两半）。
  以前有 `all` 一次给七张，七张 PNG 的 base64 比数字大两个数量级，删了。
  缺省给 `level` 是因为它按窗口算（认 `seconds`）、27KB、0.8s，最便宜也最没有歧义；
  `spectrogram` 一张 384KB，当默认等于替调用方做了个很贵的决定。走了默认会在 `note` 里写一句。
  `spectrum` / `wave` / `pitch` 只读**一个 100ms 的帧**，所以只认 `time`，
  传 `seconds` 会被忽略并在 `note` 里说明
- `subtitles`：`mode=ocr` 识字幕（`band` 可指范围）/ `mode=stt` 语音转写
  （silero-vad 自动断句 + SenseVoice，模型路径由环境变量指定）

音频图是正方形，边长跟着 `detail` 走：low=768（默认）medium=1024 high=1280 max=1536；
双声道视图上半 L、下半 R；`pitch` / `history` 与其它视图一样分声道（L 上 R 下）；真单声道音轨只画一张。
读数口径见「音频图的读数口径」。

视频固定在工作区 `workspace/<文件夹名>/`。
`video` / `audio` / `subtitles` 默认作用于最近打开的那个，可用 `video_id` 传文件夹名指定。

所有产物落到 `workspace/<标题>/{video,audio,subtitles}/`，可直接用文件系统管理（`ls`/`rm`/`cp`）；
`.mvv-cache/` 现在只放链接下载的临时文件（`downloads/`），不再当产物缓存。

## OCR

PP-OCRv4（det + rec）在 **numpy + onnxruntime** 上从零实现，不依赖 OpenCV / Pillow / pyclipper / ImageMagick。
读图用 ffmpeg 解 raw RGB；连通域、CTC 解码都是纯 Python。

```sh
pkg install python-onnxruntime
# 先按 INSTALL.md 3.1 把 PP-OCRv4 三个文件放进 models/（约 16MB）
python3 ocr.py out/ocr_test.png      # 单图
```

- 加速器：默认 `CPU`；可用 `MVV_ORT_PROVIDERS` 指定 `xnnpack` / `nnapi`（实测 CPU 最快，见「性能」）
- 调参：`DET_MAX_SIDE`（默认 720）、`DET_TH`（默认 0.2）、`MVV_ORT_THREADS`
- 实测：880x320 三行中英混排，1.0s 识别完毕，置信度 0.94–0.98

## 取窗：先选时间戳，再选看多久

所有耗时动作（`video` / `audio` / `subtitles`）共用一套取窗：

| 参数 | 含义 |
|---|---|
| `time` | 从哪个时间戳开始，默认 0（frame 里是第一帧的时刻） |
| `seconds` | 看多久；不传则用该工具的默认值 |

各工具的范围：`video` / `subtitles` / `audio` **1000 秒**（约 17 分钟），
不传 `seconds` 默认看 100 秒。

超过上限会被夹住，并在返回里带 `note` 说明夹到了哪里。长视频靠 `time` 翻页，
不会一次跑到底。

`video` 一页的代价是「页」，不是「秒」：`quantity` 格铺在多大的窗口上，页的大小都一样。
需要通读长片就按上限翻页（`time=0`、`1000`、`2000`…），只看某一段就把窗口收窄，
那时再把 `quantity` 提到 36。

`subtitles` 同理：OCR 一秒一帧，但**跳帧**会把字幕带没变的帧整段跳过（`BV1Rx54znE7k` 上实测
跳过 160/328，约一半），STT 比实时快。

这条只写进两个工具的 `description`，**不塞进返回**。返回里的 `note` 只说这一次
调用发生了什么（被夹住、到结尾、跳了几帧），不兼职劝告——每轮重复一句「你该读
大一点」，既占 token 又像在教训人。规矩写在工具定义里，`description` 每次调用
都在，不缺提示的机会。

`audio` 也能开长窗口，但一张图是定宽的：窗口越长，同样的像素铺得越开、越糊。
要读细就把窗口收窄，或者把 `detail` 提到 medium / high / max。

```jsonc
{"mode":"stt","time":1200,"seconds":600}   // 从 20 分开始，转 10 分钟
```

## 音频图的读数口径

`level` / `spectrum` / `spectrogram` / `peak` / `wave` 五张图共用一套标定，
跨窗口、跨文件都能横向比；`pitch` 与 `history` 按声道各测，不适用「L 上 R 下共用标定」的说明；表内是单声道口径。

| 项 | 口径 |
|---|---|
| FFT 点数 | 按采样率折算 100ms（44.1k → 4410 点，分辨率 10.0 Hz；48k → 4800 点，10.0 Hz）；`frame_ms` 可改 5–200，分辨率 = sr / 块长 |
| 幅度 | 满幅正弦 = 1（`2·|X| / sum(win)`），不按窗口峰值归一 |
| 峰值频率 | 取自 `time` 处那一个 100ms 帧（不是整段窗口的均值），对数域抛物线插值，幅度再除以汉宁窗在偏移处的增益（扇贝损失）|
| 电平 | 20 ms 不重叠帧的 RMS，单位 dBFS；`frame_ms` 可改 5–200（只影响曲线的粗细，不改口径）|
| 纵轴 / 色标范围 | 顶端 = 峰值**向上取整到 10 的整数次幂**，底端退 3~6 个十进位档 |
| 时频图峰值格 | 每列单独补扇贝损失，纯音不论落在 bin 哪个位置都读准 |
| 基频轨迹 | 20 ms 块逐帧测基频（`frame_ms` 可调 5–200），默认 20 ms 一帧一进，15 s 窗口 750 点；20 ms 正好容纳 50 Hz 的一个整周期，下限就是搜索下限 50 Hz（更短的帧按 1000/`frame_ms` 抬高）。块仍偏短，只给趋势。**按声道各出一份**，图仍是上下两幅（上频率对数轴 100–5000 Hz、下周期线性轴）；用散点不连线，帧间跳变太碎；周期轴按 P98 定标，个别低八度跳点贴顶不压扁全图 |
| 基频 | **按声道各测**（L / R 各自的归一化自相关，帧长 `frame_ms` 默认 100ms，下限自动抬到 1000/`frame_ms` Hz）；**只读 `time` 处那一帧**——窗口给多长都不改变读数，两个声道读的是同一个起点，天然同一时刻。不加窗（55Hz 上加汉宁窗会偏 21 音分）；**峰挑选用「数峰法」**——只留超过全局最大一半的局部极大，周期 = 最后一个峰位置 ÷ 峰数（比单取第一个峰稳，谐波强时不会被倍频骗走）。搜索范围 50–5000 Hz。置信度 = 峰值处相关系数。**为什么不混成单声道**：两路装不同声音时（双耳素材、左右不同音轨），混合的自相关会把两条周期的公倍当成基频，报出一个两边都没有的音——实测本表素材 t=53.5s，L=266.9Hz C4、R=375.0Hz F#4，混合却报 320.9Hz E4（−46.7 音分）。mono 音轨 ffmpeg 把同一路复制到两边，两边读数相同，只报 left |

频域图横轴对数 10 Hz–Nyquist，时频图 20 Hz–Nyquist，波形取的就是那 100ms 帧。

**与 phyphox 的关系**：`pitch` 的方法（100ms 块 / 半数最大幅度门限 / 最后一个极大 ÷ 极大数量）
参考 phyphox 的 *Audio Autocorrelation*、*Frequency History*、*Audio Amplitude* 三篇**实验文档**
（概念层面）。phyphox 的 App 以 **GPL-3.0** 发布，本项目**没有复用它的任何源码**，
只依据文档描述独立实现；`phyphox` 与 `RWTH Aachen` 是 RWTH Aachen University 的注册商标，
本项目与之无隶属或背书关系。

有意偏离的一处：本工具的自相关**做了能量归一化**（除以重叠段能量），phyphox 不归一化、
靠相对阈值找峰。归一化是为了跨窗口 / 跨文件横向比读数，代价是 `curve` 的纵轴不再等同于
phyphox 界面上的相对值。

已知偏差（有意保留）：时频图的噪声底比读数低 1.76 dB，这是纯音标定 + 汉宁窗
ENBW=1.5 的必然结果；全局减 1.76 dB 会让纯音偏低，所以不动。

## 输出不裁剪、也不美化

`_dump()` 默认**不截断**。之前 8000 字符上限会把 97 段字幕砍成 55 段还不显眼，
现在整份返回；确实要塞小响应时才显式传 `limit`。

返回值也一律是紧凑 JSON（无缩进、分隔符不带空格）：读者是模型不是人，
缩进和空格不提供任何信息，只按字符吃 token。一条 62 点的峰值轨迹缩进后
每个 `[t, f]` 占 6 行、字符翻一倍多。要看就 `python3 -m json.tool` 过一遍。

字幕段统一是 `{"from": "MM:SS", "to": …, "text": …}`，OCR 与语音转写同形，
折叠后整份返回；要 SRT 直接拿三个字段拼，工具本身不落盘。

**时间只到秒。** 毫秒对定位没用——没人会拿 `00:23.417` 去切帧——
但每段两个时间各写 4 个字符，98 段字幕就是 784 字符（约 7% 的响应）。
OCR 的采样步长本来就是 1.0s，毫秒位恒为 `000`，留着等于伪造精度。

### 两条路能拿到的东西不一样

| | `mode=ocr` | `mode=stt` |
|---|---|---|
| 读的是 | **画面像素** | **声音波形** |
| `text` | 屏幕上的字 | 嘴里说的话 |
| `conf` | 有（OCR 置信度） | 无 |
| `emotion` | **没有** | 有 |
| `event`（BGM/Applause/Laughter） | **没有** | 有 |
| `lang` | **没有** | 有 |
| `summary` 统计 | 无 | 有 |

**声学三项只有 stt 有，这不是没实现，是原理上取不到**——
OCR 只看像素，画面里没有「情感」和「有没有 BGM」这两样信息。

这条事实写在工具 schema 的 `mode` 描述里，**不写进每次响应**——
调用方自己传的哪个 mode，它知道；重复声明等于每次为已知信息付费。

所以：

- 想知道 **BGM / 语气 / 语种** → 必须走 `mode=stt`，即使字幕已经在画面上了
- 想知道 **画面上的字**（歌词、烧录字幕、弹幕）→ 走 `mode=ocr`
- 两样都要 → 调两次。这类片子通常有口播，先跑 stt 拿语义，再用 ocr 核对画面原文

语音转写在 stt 模式下会带上 SenseVoice 同一次前向算出的三项（有才写，默认值省略）：

| 字段 | 取值 | 说明 |
|---|---|---|
| `emotion` | `HAPPY` / `SAD` / `ANGRY` / `NEUTRAL` … | `NEUTRAL` 与 `EMO_UNKNOWN` 不写 |
| `event` | `Speech` / `BGM` / `Applause` / `Laughter` … | `Speech` 不写 |
| `lang` | `zh` / `en` / `ja` / `ko` / `yue` … | 这首歌里出现 `en` 就是混了英文句 |

顶层另给一个 `summary` 统计（如 `{"event": {"BGM": 4}}`）。
`event=BGM` 说明那段里音乐压过了人声，转写掉字多半就是这个原因——
但注意它是模型对整段音频的判断，不是声道分离的结论。

## 性能

实测环境：Android aarch64，Python 3.14，onnxruntime 1.31.1。

**线程**：这机器 **2 线程最优**。det 推理 1 线程 0.182s → 2 线程 **0.117s** → 4 线程 0.125s →
8 线程 0.149s（小核争用）。`MVV_ORT_THREADS` 默认 2。

ffmpeg 解码分开管：视频 / 音频工具 `MVV_FFMPEG_THREADS=3`（单跑比 2 条快 12%，
8 条只再多 9%），字幕取帧 `MVV_OCR_FFMPEG_THREADS=2` × 2 进程。为什么不是一个数，
见「踩坑记录」1。

**加速器**：`CPU` 实测最快，`XNNPACK` 反而更慢，`NNAPI` 初始化贵。默认 `cpu`，可用 `MVV_ORT_PROVIDERS` 改。

**字幕带 + 跳帧**（`subtitles` 的 `mode=ocr`）是最有效的一刀。

30 秒测试片（15 条字幕，每条 2 秒）：

| 方案 | 采样 | 总帧数 | OCR 次数 | 耗时 |
|---|---|---|---|---|
| 全帧逐帧 | 1.0s | 30 | 30 | 8.10s |
| **字幕带+跳帧** | **1.0s** | **30** | **15** | **2.20s** |

原理：只裁一条字幕带（不是整个下半屏），字幕没变的帧直接复用上次结果。
带高由自动检测给出，1280×720 上实测 84–129px，占整幅 12–18%。

**字幕带是怎么定的**（`mode=ocr` 且 `band=auto` 时）：

- 两段式：先只 OCR 下方 40%（多数片子的字幕在那儿，只花 0.4 屏）；找不到够像的簇，
  再补扫上半屏。两段共用同一批已解码的帧，整幅只扫一遍、不重复解码
- 簇按纵向重叠聚类，并限高 25%：重叠是传递的，一屏小字能从顶串到底，
  串成一个 0.0–0.8 的「簇」会让选带彻底跑偏（实测栽过一次）
- 8 个采样点按时长均分；窗口 ≤45 秒走一次解码取帧，更长的逐点 seek（成本与片长无关）
- 先筛掉不像字幕的簇：最宽 <12%（单个字母/图标）、纵向串成柱 >25%（几个无关的框连成一片）
- 簇打分 = 行高中位数^1.5 × 宽度 × 位置权重 × 出现次数；位置权重给底部（1.0→2.2）和
  顶部（1.0→1.6）加分，画面中部压到 0.6
- 纵向间隔 ≤2% 的簇并成一条带（并完不超过画面 1/5）
- 边距按实测行高给（行高的 0.6 倍，至少 0.02）：两行字幕会整块上移，固定边距会切掉一行
- 常驻水印：同一串字（去标点、大小写归一）在簇内占比 ≥50% 判为静态标，得分 ×0.25
- 一点字都没检出就判「无字幕」，直接返回空；`band=auto` 且 0 段时返回 `hint`

**注意：耗时由「字幕条数」决定，不由采样密度决定。** 同一测试片：

| 采样 | 总帧数 | OCR 次数 | 耗时 |
|---|---|---|---|
| 0.5s | 60 | 15 | 2.33s |
| **1.0s（默认）** | **30** | **15** | **2.29s** |
| 1.5s | 20 | 15 | 2.24s |

采样越密只多出便宜的跳帧比对。**默认 1.0s** 是为了省临时数据量，而非省时间：

| 视频长度 | step=0.5s | step=1.0s | step=1.5s |
|---|---|---|---|
| 30 分钟 | 691 MB | 346 MB | 230 MB |
| 1 小时 | 1382 MB | 691 MB | 461 MB |
| 2 小时 | 2765 MB | 1382 MB | 922 MB |

因此实现上**不落盘**：ffmpeg 用管道（`-f rawvideo -`）流出带帧，逐帧读、逐帧判、逐帧丢，
磁盘占用恒为零，任意长度视频都不爆盘。

**关于自量化**（结论：死路，已验证）：
权重 int8 量化（QDQ）与真·动态量化（`DynamicQuantizeLinear` + `QLinearConv`，62 个卷积）
两条路都跑通了——需自建 `onnx_pb2`，因为 Termux 装不了 `onnx` 包
（`cp314t`+manylinux 轮子对不上 `android/Bionic`）。
但 ORT 的 ARM int8 卷积**和 fp32 NEON 一样快**：推理 0.101s vs 0.101s，**提速 1.00x**，
还多出 62×3 个节点、加载慢 3 倍、精度下降（per-tensor，mean diff 0.064）。
量化工具与产物已删除，结论记录在此。

## 实测素材

参数是在这些片子上试出来的。**只列出处，不随仓库分发任何音视频**，版权归各 up 主——
谢谢他们把片子放在那儿，让这些数字有地方可测。

| 片子 | 时长 | 画面 | 拿来验什么 |
|---|---|---|---|
| [BV1NEW16wEGN](https://www.bilibili.com/video/BV1NEW16wEGN) | 654s | 1280×720，字幕带 y0 0.79–0.83 | 两档 ffmpeg 线程、1:2×2 配置、桶取帧去重、整页豁免、缓存清理 |
| [BV1Rx54znE7k](https://www.bilibili.com/video/BV1Rx54znE7k) | 328s | 1280×720，字幕逐句烧录，尾短语有独立短 cue | 跳帧阈值、短 cue 是不是噪声、ASR 预处理 |
| [BV1tx4b6AE6H](https://www.bilibili.com/video/BV1tx4b6AE6H) | 320s | 1280×720，字幕两档高度 | 字幕带切掉第二行、片头标题丢失 |
| [BV1FRao6yESc](https://www.bilibili.com/video/BV1FRao6yESc) | 276s | 1280×720，字幕贴画面顶部 | 声学字段、BGM 段掉字 |
| [BV1tWng6FE2Q](https://www.bilibili.com/video/BV1tWng6FE2Q) | 266s | 1024×720，整页文字 + 底部单行 | 整页文字判据（簇数 / 跨度） |
| [BV1rx4n6FEyg](https://www.bilibili.com/video/BV1rx4n6FEyg) | 164s | 1280×720，整页文字切换 | 整页文字、按画面变化取帧 |
| [BV1Yasf6XEj1](https://www.bilibili.com/video/BV1Yasf6XEj1) | 25s | 1080×1920，竖屏 | 分辨率判断（短边 vs 长边）、竖屏网格与整页预算 |
| [BV1FUrNBnE9y](https://www.bilibili.com/video/BV1FUrNBnE9y) | — | 立体声，左右声道不同音高，帧内换音 | 基频按声道各测、`seconds` 对单帧视图无意义 |

文中只写结论、不写片名的地方，指的就是这张表里的某一支；描述（时长 / 分辨率 / 字幕形态）
留在正文里，够复现判断。

## 踩坑记录（按改动顺序，细节看提交）

每条都记下当时的**错误判断**和把它推翻的那次实测。数字都能用 `scripts/` 里的基准脚本复现。

**1. ffmpeg 的 `threads=auto` 在小核上抢光 CPU**

一条开关管所有 ffmpeg 是不对的：视频 / 音频工具同一时间只跑一路，字幕却默认 2 进程并行。
现在分开：工具 `MVV_FFMPEG_THREADS=3`，字幕 `MVV_OCR_FFMPEG_THREADS=2`。

| 用途 | 默认 | 为什么 |
|---|---|---|
| 视频 / 音频工具 | **3** | 只跑一路，多给一条不挤别人 |
| 字幕取帧 | **2** | 2 进程并行，每进程 2 条 = 4 条在跑 |

`scene_scores` 0-200s（每种各跑两遍）：2 条 10.80s → **3 条 9.48s** → 8 条 8.69s。
3 条拿走九成收益，剩下 5 个核留给同时跑的 OCR。字幕那边是「1:2 一份、复制两份」：
`MVV_OCR_WORKERS=2` 个进程，每进程 1 条 ORT 线程 + 2 条解码线程。

| 配置（90 秒窗口） | 单独跑 | 与 `scene_scores` 600s 并发 |
|---|---|---|
| 串行 1 进程：ORT 2 + 解码 2 | 16.3 / 19.0s | OCR 32.3s，扫描 57.7s |
| **2 进程 ×（ORT 1 + 解码 2）＝默认** | **15.1 / 15.4s** | **OCR 23.0s**，扫描 57.7s |
| 2 进程 ×（ORT 2 + 解码 2） | 16.5s | — |
| 3 进程 / 4 进程 | 21.3 / 18.8s | — |
| 单进程：ORT 1 + 解码 2 | 19.6s | — |

最后一行是关键：**1:2 只有配上第二份才成立**，单进程里把 ORT 砍到 1 条是纯亏。

三个坑：**第一次跑的数字不能当基线**（9 格一开始测成 8.18s vs 0.97s，其实是第二次命中了
帧图 `t*.jpg` 和 `.scene-<hash>.json`，压根没解码）；**`-threads` 是管下一个输入的**
（36 张缩略图拼成一条命令那条，得在每个 `-i` 前各写一个）；**量进程池的脚本必须有
`if __name__ == "__main__"`**（否则 spawn / forkserver 的子进程会把脚本再跑一遍，
我这样量出过「并行慢一倍」的假结论）。

顺带：`ocr.py` 原来那个 `MVV_FFMPEG_THREADS` 是死开关——定义了，但 `_stream_frames`
把 `2` 写死了，环境变量等于假的。

**2. 选帧：「变化最猛」≠「值得看的格子」**

老办法按分数从高到低取、NMS 拉开间距、不够就插值补齐。654s 全片 9 格里 **6 格是同一个
主持人镜头**——分数高的都挤在他讲话那段。改成三层：按时间均分 n 桶、每桶留 3 个候选；
桶内首选太暗（均值 <16）或纯色（标准差 <4）换下一个；与已选帧 dHash 差 ≤10 位换下一个。
桶内全灭才退回首选：宁可重复一格，也不空着。

效果用哈希统计（与 28.24s 那个主持人镜头差 ≤12 位算同场）：全片 9 格 6 → **2 格**，
0-200s 9 格 3 → 2，36 格 6 → 5。代价：9 格 3.6 → 5.3s、36 格 16.0 → 16.9s，多出来的
都是「换人」重抽的帧；`MVV_BUCKET_K=1` 等于关掉换人。

坑：桶中点写成 `span*(i+0.5)`、漏了 `/n`（n=9 时中点跑到 5.0 / 15.0 / 25.0，被测试逮住）；
docstring 里我先写了「取点法能差 20 多位、块均值只差 2-9 位」，一测全是编的（同镜头
3-6 位 vs 3-4 位，换编码漂移 0.92 vs 1.08 位）；13 张缩略图一个进程一个 spawn 要 1.16s，
改成一条 `hstack` 命令后 9 张 0.60 → 0.11s，逐字节相同。

**3. 整页误判：抬阈值是凑巧，分辨要看那条带**

`_page_like` 判「这一屏是不是整页文字」，原判据「可用簇 ≥4 且跨度 ≥0.45」。柴知道这支片
6 个 100s 窗口**全部**被判整页，可字幕读得好好的（61 / 57 / 56 / 60 / 54 段）。根因是
台标和图表标签只有 0.13-0.20 宽，但随镜头换高度，凑出 4-5 个簇、跨度 0.70-0.87。

第一次改法把门槛抬到 6，6 个窗口全变 False——但 200-300 那窗也变 False 了，它读出来
11 段**全是图注**（`USB-2.0`、`640x480(?)`），本来就该报整页。抬阈值只是让样本碰巧分开。

正解看选中的带自己：`y0 ≥ 0.6` 且 `w ≥ 0.30` 且 `count ≥ 采样数/2` 就判字幕——
屏幕上别处的字只是内容。实测分寸正好卡在 0.6（真字幕带 y0 = 0.787-0.830，图注带 0.559）。
第二版还有个坑：豁免去簇列表里找「与带重叠的那一簇」用它的 w / count，500-600 那窗漏了
——合并相邻行之后两者不一致，改成直接用 band 自己的统计。

**4. 整页判定不能靠单帧**

960 宽的预览图上，一页论文只解出 3 个簇、宽度全在 0.13-0.14；主持人 + 字幕那帧 3 个簇、
宽 `[0.14, 0.21, 0.49]`。低分辨率下单帧永远凑不齐「整页」的特征，只能靠多帧聚合。

**5. 下载缓存只做中转**

`.mvv-cache/downloads` 原来是只进不出：下载件拷进工作区后还留着，一支 51MB 的片占两份，
`MVV_KEEP_WORKSPACES` 又只管工作区。现在拷完就删（大小对得上才删），无主的一小时后扫掉。

**6. 别的都写在提交里**：`_prune_workspaces` 曾定义了没人调（`MVV_KEEP_WORKSPACES` 等于
假开关）；字幕带检测的 0.45 兜底会把画面里的大字当字幕；行高基准从中位数回退到 `max`
（中位数多留下的恰恰是游戏 UI 边缘文字）；带内彩色像素抹黑再识别修好一例红心、
却丢掉整行颜色分界，撤了。

## 协议与安全

### MCP 协议细节

- **协议**：JSON-RPC 2.0；`initialize` 回 `protocolVersion: 2024-11-05` 和 `serverInfo.name = mcp-vision`。
- **能力**：只暴露 `tools`（`capabilities: {"tools": {}}`），**没有** `resources` / `prompts` / `sampling`。
  工具四件套：`workspace` / `video` / `audio` / `subtitles`（见「工具」），外加一个 `ping`。
- **传输**：
  - **stdio**（默认）：`python3 mcp_server.py`，一行一个 JSON-RPC 请求，响应一行；
  - **HTTP**：`python3 mcp_server.py --http 8765`，POST JSON-RPC 到任意路径、回 JSON（没有 id 的通知回 204）。
    `--host` 默认 `127.0.0.1`，改成 `0.0.0.0` 就真的对外监听。
- **鉴权**：**没有**。没有 token，不校验 `Authorization`，stdio 与 HTTP 一视同仁——
  安全性完全建立在“只有本机进程能连上这个端口”这一个假设上（见下）。
- **返回形态**：`content` 数组，第一个元素是 `{"type": "text", "text": "<紧凑 JSON>"}`，图片跟在后面：
  视频帧是 `image/jpeg`，音频图是 `image/png`。
- **错误分两层**：
  - 输入问题（`seconds` 非正、`view` 不认识、crop 越界、没有音轨）→ 返回 `{"ok": false, "error": ...}` 文本，
    调用方读得到原因、还能继续往下走；
  - 协议层问题 → JSON-RPC `error`：请求不合法 `-32600`、JSON 解析失败 `-32700`、未知方法 `-32601`、
    未知工具 `-32602`、handler 抛异常 `-32000`。
- **窗口范围**：`video` / `subtitles` 单次最多 1000 秒（`MAX_ANALYZE_SECONDS`），不传 `seconds` 默认 100 秒
  （`DEFAULT_ANALYZE_SECONDS`）；`audio` 的窗口视图同样 1000 秒（默认 100），
  而 `spectrum` / `wave` / `pitch` 只读单帧、不认 `seconds`。长片靠 `time=0, 1000, 2000 …` 翻页，没有别的分段开关。
- **扫描缓存**：`video` 的 scene 扫描按窗口写到 `workspace/<标题>/video/.scene-<hash>.json`，
  同一段重复调用不重扫；缓存损坏时静默重扫。

### 安全边界

- **网络**：默认只绑 `127.0.0.1`，无鉴权、无 TLS。用了 `--host` 就是自己的选择——
  要对外暴露请配上反向代理和认证，否则任何连上来的进程都能读本机磁盘上的任意视频。
- **文件系统**：
  - 工作区根为 `repo/workspace/`，产物按标题分目录；
  - `workspace.open` 的相对路径按仓库根 resolve；**绝对路径不设限，`..` 也不拦**——
    这是给本机模型用的工具，不是沙箱。真要限定范围，得在外层容器里做；
  - 下载先落到 `.mvv-cache/downloads/`，再复制进 `workspace/<标题>/`，拷完就删；
    没人认领的缓存文件一小时后扫掉（宽限留给正在下载的另一路）；
  - ffmpeg 的 `crop` 在 Python 侧先解析成 `W:H:X:Y`、与帧尺寸比对，越界直接报错拒绝，
    避免 ffmpeg 不报错只吐一张黑图。
- **子进程**：ffmpeg / ffprobe / yt-dlp 全部用**参数列表**调用（`subprocess.run([...])`），
  全仓库没有 `shell=True`，用户输入进不了命令行字符串。
- **模型加载**：OCR / ASR 模型按需懒加载，启动时不碰推理库；纯静音视频 0.17 秒结束，不加载 229MB 识别模型。
  模型放 `models/`，其余路径由 `MVV_OCR_*` / `MVV_ASR_*` 环境变量指定，不随仓库分发。
- **下载授权**：`workspace.open` 可拉任意 yt-dlp 支持的 URL；**本项目不代为判断授权**，
  请自行确认对目标内容有下载 / 使用授权并遵守站点条款与当地法律。
- **无沙箱**：ffmpeg 解码不可信媒体时的漏洞风险由 ffmpeg 本身承担；本工具不对输入做额外沙箱。
- **并发**：stdio 单线程，一个请求跑完才接下一个；HTTP 每请求一线程（`ThreadingMixIn`），
  但每个工具各有一把锁——同名工具排队，不同工具可以真并行。
  一次 1000 秒的 scene 扫描只会挡住别的 `video` 调用，不挡 `audio`。

## 测试矩阵

CI（`.github/workflows/ci.yml`）跑在 **ubuntu-latest / Python 3.10 + 3.13**，装 ffmpeg + numpy，分三步：
模块可导入 → `tools/list` 冒烟（断言四个工具的名字与顺序）→ `sh check.sh --test`。

| 测试文件 | 行数 | 覆盖 |
|---|---|---|
| `tests/test_windows.py` | 500 | 参数校验与取窗（`time` 越界回退、`seconds` 上限夹取、单帧路径）、`_tc` 时间格式、JSON-RPC 错误码、`_crop_box` 解析、`_dump` 紧凑 JSON、按 `view` 裁剪字段、`_scene_buckets` 分桶与空桶取中点、拼页像素预算与画面比例、`_prune_workspaces` 按 `opened_at` 清理、输出不回显绝对路径 / 死字段、schema 条件必填、基频与频谱数值精度（440 / 1000 / 1975 / 4500 Hz、音分、置信度）、`frame_ms` 与 20/25/10 ms 的步进和下限 |
| `tests/test_page_like.py` | 121 | OCR 框聚类、`page_like` 整页判定（单行字幕 / 整页文字 / 多簇集中 / 台标簇不算整页）、窄框剔除、"带本身就是字幕"豁免的四个反例 |
| `tests/test_stereo.py` | 58 | 合成 L440 / R3000 立体声：频谱峰值左右不串、单声道标得出来、基频按声道各测且不受窗口长度影响、七张图尺寸 768×768 |
| `tests/test_keep_wide.py` | 53 | 字幕行筛选 `_keep_wide`：窄句不被 `top*0.5` 当噪音丢掉；超宽离群框不抬高宽度标尺；行高基准是 `max` |
| `tests/test_ff_threads.py` | 116 | `engine.py` 7 条 ffmpeg 命令（含 36 张批量那条）都带 `-threads`，**每个 `-i` 前各有一条**；工具默认 3 条、字幕默认 2 条，两条开关不同名；`ocr._stream_frames`、`_collect_sample_frames` 的兜底都没写死 2 |
| `tests/test_pick_frames.py` | 135 | 选帧三道筛子：分桶按时间归位 / 桶内按分数降序 / 噪声不算变化点 / 空桶取中点 / 候选数封顶；黑帧、纯色、近似帧换桶内次优；界限外不误杀；桶内全灭退回首选 |

**覆盖缺口（明确列出，不假装有）**：

- 无真实媒体的端到端测试（所有测试用合成数据 / 几何）；
- 无 scene 检测质量基准（F1 / 召回），只测了“补位不越界、不重复、不冒充满额”；
- 无 OCR / STT 精度测试（依赖模型，CI 不装）；
- 无 HTTP 传输的集成测试。

## 试过又撤掉的（避免重复踩坑）

OCR 识别器选型试过三轮，每一轮都 revert 了：

1. **支持 PP-OCRv4_server_rec_doc**——它自动跟随 doc 字典，印刷和手写都更准，
   但模型体积与加载成本压不住，撤；
2. **存在 server_rec_doc 就默认用它**——理由同上，撤；
3. **两级识别：mobile 快跑，置信度 < 0.90 才用 server_doc 复核**——准确率确实涨，
   但链路复杂度和收益不成比例，撤。

结论：最终只保留 mobile 一套。`MVV_OCR_MODE` 仍可手动切，但**默认不自动升级**——
宁可慢一点、结果一致，也不要“有时候更准、有时候不”，那会让用户没法建立预期。

另外两处也撤了：

4. **带内彩色像素抹黑再识别**（`cb82f82` → `1503800` 回退）。起因是红心被读成 `awwww`；
   抹黑确实修好了这一例，但彩色字幕整行会丢掉颜色分界，错的比对的还多，撤。
   现在去彩色阈值只作为可选项留着（`2111328` 记下了 60 → 64 的实测原因），默认不开。
5. **行高基准改中位数**（`fcc1258` → `21f05ae` 回退），理由见「行为修正」第二条。

## 状态

v0.4 已通：`workspace`（open / list）、`video`、
`audio`（level / spectrum / spectrogram / peak / wave / pitch / history）、`subtitles`（ocr / stt）。

转写链路：ffmpeg 直出 16k 单声道 → silero-vad 自动断句（阈值/静音/最长段交给模型，
不写死时间网格）→ 逐段 SenseVoice 解码 → 丢弃纯语气词片段。

当前默认值（改参数前先看这张表，别照旧文档抄）：

| 项 | 默认 | 上限 / 范围 |
|---|---|---|
| 单次窗口 `seconds` | 100 秒 | 1000 秒 |
| 一页帧数 `quantity` | 9 | 36 |
| `detail` 像素阶梯 | low（768） | medium 1024 / high 1280 / max 1536 |
| 音频 `history` 轨迹点 | — | 2000 |
| OCR 并行阈值 | 30 秒起走双进程 | 进程池起不来或子进程崩了回退串行 |

长窗口（900–1000 秒）的字幕 OCR 已从 119 秒压到 95 秒，内容逐字不变。

模型全部按需加载，启动时不碰任何推理库：

| 时刻 | 常驻内存 |
|---|---|
| 进程启动 | ~36 MB |
| 首次 OCR（只建检测模型） | ~68 MB |
| 首次框出文字（才建识别模型） | ~90 MB |
| 转写（VAD 切出第一段语音才建 SenseVoice） | ~340 MB |

纯静音视频转写 0.17 秒结束，不加载 229 MB 的识别模型。

未做：B 站元数据/评论/弹幕、镜头切分、多站登录态。

## 致谢

思路参考了以下三个项目。**代码没有复用，全部按 Python 重写**，运行时也不依赖它们。

- [video-understanding-kit](https://github.com/yangjili6/video-understanding-kit) — 把视频拆成 `analyze`（整片）和 `frame`（单帧）两层的做法，以及网格故事板的排版思路。
- [dsh-video-vision](https://github.com/Tests4gptplay/dsh-video-vision) — 链接规范化（`lib/ytdlp.js` 的 `normalizeSource`）、按时间点取原帧。
- [dsh-bili-watch](https://github.com/ppppiii/dsh-bili-watch) — B 站下载与登录态（cookies）处理的经验。

也感谢 ffmpeg、yt-dlp、numpy、onnxruntime 的作者，以及「实测素材」里那几支片子的作者——
所有参数都是在它们的画面上试出来的。

模型与依赖：

- **PP-OCRv4**，来自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0）
- **SenseVoice**，来自 [FunAudioLLM / FunASR](https://github.com/FunAudioLLM/SenseVoice)（© 阿里巴巴集团）。
  代码 MIT；**模型权重是另一套**：FunASR Model Open Source License v1.1，须保留模型名与署名
- **Silero VAD**（MIT）、**sherpa-onnx**（Apache-2.0）

## 许可证

**定位**：仅供学习研究使用。

本仓库代码：MIT，见 [LICENSE](LICENSE)。© 2026 Huangjiang-a11y

**LICENSE 只覆盖本仓库的代码。** 模型权重不随仓库分发，遵循各自的上游许可；
ffmpeg 是以子进程调用的外部程序（不链接、不捆绑）；
经由本工具处理的音视频与字幕，版权归原权利人。

各组件许可、ffmpeg 的分发注意事项、模型署名的要求，见 **[THIRD_PARTY.md](THIRD_PARTY.md)**。
