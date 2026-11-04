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

单一工具 `video_vision`，action 分档：`open` / `watch` / `frames` / `sheet` /
`audio` / `transcribe` / `transcript` / `text` / `comments` / `danmaku` /
`search` / `hot` / `status` / `clean` / `export`。

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

## 状态

v0.1 已通：probe / frames / sheet / audio / ocr / status / clean。
待做：transcribe（sherpa-onnx）、B 站元数据/评论/弹幕、TransNetV2 镜头检测、多站登录态。
