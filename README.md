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

## 状态

v0.1 已通：probe / frames / sheet / audio / ocr / status / clean。
待做：transcribe（sherpa-onnx）、B 站元数据/评论/弹幕、TransNetV2 镜头检测、多站登录态。
