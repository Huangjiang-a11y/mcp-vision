# mcp-video-vision

**让 AI 看视频、听视频、读视频。** Python 实现的 MCP 服务器，**不依赖 Node**。

把视频变成模型能消费的结构化内容：带时间戳的帧、音频信号分析、语音转写、字幕、B 站评论/弹幕、九宫格故事板。所有处理在本机完成。

## 设计原则

- **纯 Python**：stdlib + numpy + ffmpeg + yt-dlp + sherpa-onnx。无 Node、无 Pillow、无 ImageMagick。
- **自含**：不复用任何外部仓库的代码（经验可借，代码重写）。
- **MCP 优先**：stdio + HTTP 双入口，工具 schema 与 `mcp_server.py` 风格一致。
- **MIT**。

## 依赖

- Python 3.10+（开发在 3.14）
- `ffmpeg` / `ffprobe`（PATH 或配置）
- `numpy`
- `yt-dlp`（pip，多站下载）
- `sherpa-onnx`（可选，ASR 转写）
- RapidOCR（可选，画面 OCR）

## 快速开始

```sh
python3 check.sh
python3 mcp_server.py            # stdio
python3 mcp_server.py --http 8765
```

## 工具

四个大工具，按职责分层：

- `workspace`：`open`（打开/下载）/ `list`
- `video`：`probe` / `frames` / `sheet` / `peek`（`scene` 待做）
- `audio`：`levels` / `extract`（`transcribe` / `transcript` 待做）
- `subtitles`：`ocr` / `band`（`search` 待做）

所有产物落到 `workspace/<标题>/{video,audio,subtitles}/`，可直接用文件系统管理（`ls`/`rm`/`cp`），
不再用不透明的 `.mvv-cache`。

## OCR

PP-OCRv4（det + rec）在 **numpy + onnxruntime** 上从零实现，不依赖 OpenCV / Pillow / pyclipper / ImageMagick。
读图用 ffmpeg 解 raw RGB；连通域、CTC 解码都是纯 Python。

```sh
pkg install python-onnxruntime
python3 scripts/fetch-models.py      # 约 16MB；必要时设 https_proxy
python3 ocr.py out/ocr_test.png      # 单图
```

- 加速器：优先 `XnnpackExecutionProvider`，可回落 `CPU`；Android 上另有 `NnapiExecutionProvider`
- 调参：`DET_MAX_SIDE`（默认 720）、`DET_TH`（默认 0.2）、`MVV_ORT_THREADS`
- 实测：880x320 三行中英混排，1.0s 识别完毕，置信度 0.94–0.98

## 性能

实测环境：Android aarch64，Python 3.14，onnxruntime 1.31.1。

**线程数**：这机器 **2 线程最优**。det 推理 1 线程 0.182s → 2 线程 **0.117s** → 4 线程 0.125s → 8 线程 0.149s（小核争用）。
默认已设为 2，可用 `MVV_ORT_THREADS` 调。

**加速器**：`CPU` 实测最快，`XNNPACK` 反而更慢，`NNAPI` 初始化贵。默认 `cpu`，可用 `MVV_ORT_PROVIDERS` 改。

**字幕带 + 跳帧**（`action=subtitles`）是最有效的一刀。

30 秒测试片（15 条字幕，每条 2 秒）：

| 方案 | 采样 | 总帧数 | OCR 次数 | 耗时 |
|---|---|---|---|---|
| 全帧逐帧 | 1.0s | 30 | 30 | 8.10s |
| **字幕带+跳帧** | **1.0s** | **30** | **15** | **2.20s** |

原理：只裁下 30% 字幕带（640x360 → 640x100，像素少 3.6 倍），且字幕没变的帧直接复用上次结果。

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

## 实测踩坑记录：《琵琶行》视频

拿 [BV1Rx54znE7k](https://b23.tv/PTpLiZU)（328s，720p，动画背景）实测，
暴露了两个真问题：

**1. 跳帧在动画背景上失效**

原按「带内像素绝对差」判重，背景一动就超阈值，跳过仅 7/328 (2%)。
改为「灰度 > 220 的高亮文字掩码差」：字幕未变 ≤0.15%，变了 ≥0.86%，
跳过率升到 160/328 (49%)。

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

## 状态

v0.1 已通：probe / frames / sheet / audio / ocr / status / clean。
待做：transcribe（sherpa-onnx）、B 站元数据/评论/弹幕、TransNetV2 镜头检测、多站登录态。
