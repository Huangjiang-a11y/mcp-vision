# mcp-vision

[![CI](https://github.com/Huangjiang-a11y/mcp-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/Huangjiang-a11y/mcp-vision/actions/workflows/ci.yml)

**让 AI 看视频。** 纯 Python 的 MCP 服务器，不依赖 Node 运行时。

仓库结构参照 [dsh-video-vision](https://github.com/Tests4gptplay/dsh-video-vision) 的分层重写：
保留同样的 `workspace` / `video` / `audio` / `subtitles` 四层与工作区目录布局，
实现全部换成 Python —— ffmpeg 解码、numpy 计算、onnxruntime 跑 PP-OCRv4，
没有浏览器、没有 Node 进程、没有 Electron。

输入本地文件或 B 站链接，输出模型能直接消费的东西：带时间戳的帧、近似正方形的网格故事板、
音频电平与静音区间、画面 OCR 与折叠后的字幕轨。全部在本机完成。

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

模型不随仓库分发，路径全部走环境变量：

| 变量 | 用途 |
|---|---|
| `MVV_OCR_DET` / `MVV_OCR_REC` / `MVV_OCR_KEYS` | PP-OCRv4 三个文件 |
| `MVV_ASR_MODEL` / `MVV_ASR_TOKENS` / `MVV_VAD_MODEL` | SenseVoice + silero VAD |

没配就只缺对应能力，抽帧和电平分析不受影响。详见 [INSTALL.md](INSTALL.md)。
- `yt-dlp`（可选，链接下载）

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
- `video`：给窗口 + 张数，抽帧拼成一页网格；`quantity=1` 时不出页，直接给单帧原图（可 `crop`）
  - `quantity` 抽帧数（默认 15，上限 30；1 = 单帧）；`quality` 像素上限：low=512² medium=1024² high=1536² max=2048²（拼页时管整页，单帧时管裁剪后那张）
- `audio`：按 `view` 出图 —— `level`（20ms RMS 电平）/ `spectrum`（FFT 幅度谱）/
  `spectrogram`（时频图）/ `peak`（峰值频率轨迹）/ `wave`（原始波形），`all` 一次全给
- `subtitles`：`mode=ocr` 识字幕（`band` 可指范围）/ `mode=stt` 语音转写
  （silero-vad 自动断句 + SenseVoice，模型路径由环境变量指定）

音频图一律 1024×1024，上半 L、下半 R。读数口径见「音频图的读数口径」。

视频固定在工作区 `workspace/<文件夹名>/`。
`video` / `audio` / `subtitles` 默认作用于最近打开的那个，可用 `video_id` 传文件夹名指定。

所有产物落到 `workspace/<标题>/{video,audio,subtitles}/`，可直接用文件系统管理（`ls`/`rm`/`cp`），
不再用不透明的 `.mvv-cache`。

## OCR

PP-OCRv4（det + rec）在 **numpy + onnxruntime** 上从零实现，不依赖 OpenCV / Pillow / pyclipper / ImageMagick。
读图用 ffmpeg 解 raw RGB；连通域、CTC 解码都是纯 Python。

```sh
pkg install python-onnxruntime
# 先按 INSTALL.md 3.1 把 PP-OCRv4 三个文件放进 models/（约 16MB）
python3 ocr.py out/ocr_test.png      # 单图
```

- 加速器：优先 `XnnpackExecutionProvider`，可回落 `CPU`；Android 上另有 `NnapiExecutionProvider`
- 调参：`DET_MAX_SIDE`（默认 720）、`DET_TH`（默认 0.2）、`MVV_ORT_THREADS`
- 实测：880x320 三行中英混排，1.0s 识别完毕，置信度 0.94–0.98

## 取窗：先选时间戳，再选看多久

所有耗时动作（`video` / `audio` / `subtitles`）共用一套取窗：

| 参数 | 含义 |
|---|---|
| `time` | 从哪个时间戳开始，默认 0（frame 里是第一帧的时刻） |
| `seconds` | 看多久；不传则用该工具的默认值 |

各工具的上限：`audio` **15 秒**（不传默认 10 秒）、`video` / `subtitles` **900 秒**（15 分钟）。

超过上限会被夹住，并在返回里带 `note` 说明夹到了哪里。长视频靠 `time` 翻页，
不会一次跑到底。

```jsonc
{"mode":"stt","time":1200,"seconds":600}   // 从 20 分开始，转 10 分钟
```

## 音频图的读数口径

五张图共用一套标定，跨窗口、跨文件都能横向比：

| 项 | 口径 |
|---|---|
| FFT 点数 | 4096；44.1k 下分辨率 **10.77 Hz**，窗长 92.9 ms |
| 幅度 | 满幅正弦 = 1（`2·|X| / sum(win)`），不按窗口峰值归一 |
| 峰值频率 | 对数域抛物线插值，幅度再除以汉宁窗在偏移处的增益（扇贝损失）|
| 电平 | 20 ms 不重叠帧的 RMS，单位 dBFS |
| 纵轴 / 色标范围 | 顶端 = 峰值**向上取整到 10 的整数次幂**，底端退 3~6 个十进位档 |
| 时频图峰值格 | 每列单独补扇贝损失，纯音不论落在 bin 哪个位置都读准 |

频域图横轴对数 10 Hz–Nyquist，时频图 20 Hz–Nyquist，波形取的就是那 4096 点。

已知偏差（有意保留）：时频图的噪声底比读数低 1.76 dB，这是纯音标定 + 汉宁窗
ENBW=1.5 的必然结果；全局减 1.76 dB 会让纯音偏低，所以不动。

## 输出不裁剪

`_dump()` 默认**不截断**。之前 8000 字符上限会把 97 段字幕砍成 55 段还不显眼，
现在整份返回；确实要塞小响应时才显式传 `limit`。

字幕段统一是 `{"from": "MM:SS.mmm", "to": …, "text": …}`，OCR 与语音转写同形，
折叠后整份返回；要 SRT 直接拿三个字段拼，工具本身不落盘。
时间精度取决于来源：OCR 精确到采样点，转写精确到毫秒——都按实际算出来的给，不凑整。

## 性能

实测环境：Android aarch64，Python 3.14，onnxruntime 1.31.1。

**线程数**：这机器 **2 线程最优**。det 推理 1 线程 0.182s → 2 线程 **0.117s** → 4 线程 0.125s → 8 线程 0.149s（小核争用）。
默认已设为 2，可用 `MVV_ORT_THREADS` 调。

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
  串成一个 0.0–0.8 的「簇」会让选带彻底跑偏（《areyoulost》实测栽过一次）
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

## 实测踩坑记录

起初两支片子：动画 MV [BV1Rx54znE7k](https://b23.tv/PTpLiZU)（328s，720p，有歌词字幕）、
实拍 vlog（320s，白色泡沫满屏）。后来又加了《areyoulost 降调》（字幕贴顶）
和《南充 0 分作文》（整屏手写稿）两支。按发现顺序记：

**1. 跳帧在动画背景上失效**

原按「带内像素绝对差」判重，背景一动就超阈值，跳过仅 7/328 (2%)。
改为「灰度 > 223 的高亮文字掩码差」：字幕未变 ≤0.15%，变了 ≥0.86%，跳过率升到 160/328 (49%)。
（220 与 223 在这几支片子上结果完全一致，取 223 只是留点余量。）

**2. 不能把「短 cue」当噪声合并（重要）**

该视频的歌词会把每句尾短语**重复演唱一遍**，字幕轨因此有独立短 cue：
`小弦切切如私语` → `如私语`、`大珠小珠落玉盘` → `落玉盘`。

最初误判为「OCR 残句」和「逐字擦除动画」并做了后缀合并——两次都错了。
查证过程：
- 密集抽帧（10fps）：75.50 还是完整 7 字，75.60 直接变成居中 3 字，**硬切**
- 剩余 3 字**重新居中**，不是右侧残留 → 是新 cue，不是擦除
- SenseVoice 转写：`…如私语，如私语。` —— **唱了两遍**，证实是真实重复

修正后 `fold_segments` 只做三件事：丢空帧、同文延长时间、编辑距离合并
同一 cue 的 OCR 变体。**「更短的纯后缀」一律视为新 cue 保留。**

**3. ASR 预处理反直觉**

试过降速/加音量/带通/降噪，**全部让识别更差**：

| 处理 | 结果 |
|---|---|
| 原始 | `下弦奇迹如私雨，如私雨。` ← 最好 |
| 0.8x + 1.2x 音量 | `雨下弦奇啼不思语，不思语。` |
| 0.7x + 2x + 带通+降噪 | `雨下弦情情不思雨，不思雨草。` |

降速破坏音高与时长特征，带通削掉辅音高频，对唱腔 + BGM 反而有害。

**4. 字幕带会切掉第二行**

320s、实拍、白色泡沫满屏。0–60 秒只出 4 段，片头标题整句丢失。

聚类打印显示这片的字幕有两档高度：大字在 0.822–0.900，矮的一行低到 0.902–0.950。
检测只取最高分那一簇，带定成 0.8022–0.92，而 t=0 那行在 0.907–0.944——
下边缘从字中间切过去，OCR 读不出来。

改为「纵向间隔 ≤2% 的簇并成一条带」（上限画面 1/5）后，带变 0.8022–0.97，
段数 4 → 15，片头标题和「说搞就搞」都回来了；《琵琶行》仍是 19 段，无回归。

代价：条子从 84px 涨到 129px，条内变化更容易越过 `skip_th`，
同一窗口 OCR 19 → 24 次、8.2 → 9.9 秒。`skip_th` 提到 1.5 可回到 21 次且段数不变，
但 3.0 会丢掉 5 段，没敢默认动。

**5. 网格凑整：要 12 帧会回 15 帧（已修）**

早期版本为了让整页接近方形，按 cols×rows 抽帧，`quantity=12` 实际返回 15 帧
（排成 3×5，填 12 格空 3 格）。现在要几张给几张，网格只决定排版，最后一排可以留空。

同时把空格惩罚从 0.05 提到 0.5：只要有无空格的方案，就不为了更方形去留黑格。
起因是 4 帧裁剪成字幕带时选了 1×5——页面比例 0.98 确实比 1×4 的 1.23 更方，
但白占一格。现在 4 帧稳定走 1×4。

**6. 整屏都是字的片子：没有可用的字幕带**

《南充 0 分作文》整屏是手写作答纸，底部压一行白色旁白。下方 40% 全是作文正文，
自动检测给出 23 段（改全帧后 37 段），条条是作文不是字幕。试过四条路：

| 判据 | 为什么不成立 |
|---|---|
| 提 `text_th` | 纸面比字幕还亮（≥250 的像素占 15–18%，字幕行只有 6%） |
| 按行高筛 | 作文行和字幕行高度同量级 |
| 亮度极性（白字黑边 vs 黑字白底） | 反了，纸底更亮 |
| 时间变化（字幕变、纸不动） | 纸在滚动，t=10 vs t=16 整屏每段差 22–31 |

结论是**结构上不适用**，不是参数没调好。`band` 手动指到 0.90–0.965 确实能读出那行
旁白，但那是人看着画面定的；自动检测给不出来。

## 状态

v0.4 已通：`workspace`（open / list）、`video`、
`audio`（level / spectrum / spectrogram / peak / wave）、`subtitles`（ocr / stt）。

转写链路：ffmpeg 直出 16k 单声道 → silero-vad 自动断句（阈值/静音/最长段交给模型，
不写死时间网格）→ 逐段 SenseVoice 解码 → 丢弃纯语气词片段。

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

也感谢 ffmpeg、yt-dlp、numpy、onnxruntime 的作者。

模型与依赖：

- **PP-OCRv4**，来自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0）
- **SenseVoice**，来自 [FunAudioLLM / FunASR](https://github.com/FunAudioLLM/SenseVoice)。
  代码 MIT；**模型权重是另一套**：FunASR Model Open Source License v1.1，须保留模型名与署名
- **Silero VAD**（MIT）、**sherpa-onnx**（Apache-2.0）

## 许可证

本仓库代码：MIT，见 [LICENSE](LICENSE)。© 2026 Huangjiang-a11y

**LICENSE 只覆盖本仓库的代码。** 模型权重不随仓库分发，遵循各自的上游许可；
ffmpeg 是以子进程调用的外部程序（不链接、不捆绑）；
经由本工具处理的音视频与字幕，版权归原权利人。

各组件许可、ffmpeg 的分发注意事项、模型署名的要求，见 **[THIRD_PARTY.md](THIRD_PARTY.md)**。
