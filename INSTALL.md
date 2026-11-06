# 安装指南

`mcp-vision` 是一个 MCP 服务器：把视频变成 AI 能读的东西（帧、网格页、音频电平、字幕、转写）。
本机跑，不上传。

> **先读免责声明**：本项目仅供个人学习研究，软件按 MIT「AS IS」提供、不作任何担保；
> 你下载的视频、字幕、弹幕、评论，版权归原作者所有。完整条款见文末。

---

## 1. 需要什么

| 项目 | 要求 | 说明 |
|---|---|---|
| Python | 3.10+（实测 3.14） | |
| ffmpeg / ffprobe | 任意较新版本 | 抽帧、音频分析、转写的解码全靠它 |
| 磁盘 | ≥ 400 MB | OCR 模型 16 MB，转写模型 229 MB，视频缓存另算 |
| 内存 | ≥ 1 GB 空闲 | 转写模型加载后常驻约 500 MB |

只做抽帧和电平分析的话，**不需要模型**，装完 ffmpeg + numpy 就能用。

## 2. 装依赖

### Android / Termux

```sh
pkg update
pkg install python ffmpeg python-numpy python-onnxruntime
pip install sherpa-onnx yt-dlp
```

`python-onnxruntime` 是 Termux 的包，不要用 pip 装（没有对应 wheel）。

### Debian / Ubuntu

```sh
sudo apt install python3 python3-pip ffmpeg
pip install numpy onnxruntime sherpa-onnx yt-dlp
```

### macOS

```sh
brew install python ffmpeg
pip install numpy onnxruntime sherpa-onnx yt-dlp
```

`yt-dlp` 只有下载链接才需要；`onnxruntime` / `sherpa-onnx` 分别只给 OCR 和转写用，缺了不影响其它功能。

## 3. 取模型

### 3.1 OCR（画面文字，约 16 MB）

```sh
cd mcp-vision
python3 scripts/fetch-models.py
```

模型落到 `models/`：`ch_PP-OCRv4_det_infer.onnx`、`ch_PP-OCRv4_rec_infer.onnx`、`ppocr_keys_v1.txt`。

国内网络慢的话设镜像：

```sh
HF_ENDPOINT=https://hf-mirror.com python3 scripts/fetch-models.py
```

### 3.2 转写（语音 → 文字，约 229 MB）

需要两个东西：**VAD**（断句）和 **SenseVoice**（识别）。

```sh
mkdir -p ~/asr/sensevoice ~/asr/vad

# VAD：silero，629 KB
curl -L -o ~/asr/vad/silero_vad.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

# 识别模型：只从压缩包里取需要的两个文件
cd ~/asr
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xjf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2 \
  sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
  sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt
mv sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx ~/asr/sensevoice/
mv sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt       ~/asr/sensevoice/
rm -rf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
```

压缩包约 1 GB（里面同时有 fp32 和 int8 两套权重），只留 int8 的 229 MB，下完记得删包。

**已经装过 `~/asr`（比如跑过语音输入服务）就不用重下**，本项目直接复用那里的模型。

不想放 `~/asr` 的话，用环境变量指过去：

```sh
export MVV_ASR_HOME=/path/to/asr   # 该目录下需要有 sensevoice/ 和 vad/
```

## 4. 启动

```sh
# stdio（给 Claude Desktop / 各类 MCP 客户端用）
python3 mcp_server.py

# HTTP（给不支持 stdio 的客户端）
python3 mcp_server.py --http 8765
```

客户端配置示例（Claude Desktop 风格）：

```json
{
  "mcpServers": {
    "vision": {
      "command": "python3",
      "args": ["/absolute/path/to/mcp-vision/mcp_server.py"]
    }
  }
}
```

必须用**绝对路径**，客户端的工作目录不一定是仓库目录。

## 5. 验证

```sh
sh check.sh
```

期望输出：

```
  [OK] python3 / ffmpeg / ffprobe / numpy / onnxruntime / yt_dlp
  [OK] sherpa_onnx
  [OK] STT 模型（~/asr）
  [OK] OCR 模型已就位
  工具: ['workspace', 'video', 'audio', 'subtitles']
```

任何一项是 `[--]`，就少了个可选能力，其余照样能跑。

冒烟测试：

```
workspace open  "file:///path/to/a.mp4"
video     analyze quantity=15
audio     transcribe start_seconds=0 end_seconds=30
```

## 6. 目录约定

```
mcp-vision/
├── models/            OCR 权重（gitignore）
├── workspace/         产物：每个视频一个文件夹（gitignore）
│   └── <标题>/
│       ├── video/     probe.json / frame_*.jpg / sheet_*.png
│       ├── audio/     levels.json / audio.wav
│       └── subtitles/ cues.json / cues.srt
├── cookies.txt        B 站登录态（gitignore，**绝不要提交**）
└── out/               实验产物（gitignore）
```

## 7. 常见问题

**`sherpa_onnx` 装不上**
Termux 上 `pkg install python-onnxruntime` 后再 `pip install sherpa-onnx`；如果 pip 编译失败，说明该平台没有预编译 wheel，改用别的设备跑转写。

**转写返回空**
先确认视频有音轨：`ffprobe -show_streams`。纯 BGM 无对白的片子，VAD 会判成「没有语音」，属正常。

**转写结果错字多**
唱腔、强 BGM、方言都会掉准确率。可以对同一段换 `language`（`zh`/`en`/`ja`/`ko`）或用更小的 `max_speech` 重试。

**OCR 报 `No module named onnxruntime`**
装 `python-onnxruntime`（Termux）或 `pip install onnxruntime`。

**下载 B 站视频 403 / 需要登录**
把浏览器 cookies 导出成 Netscape 格式存为 `cookies.txt`。该文件含账号凭证，别分享、别提交。

**抽帧很慢**
`quality` 调 `low`，`quantity` 调小；解码本身受手机 CPU 限制。

---

## 免责声明

**一、软件本身**

本项目以 MIT 许可发布，见 [LICENSE](LICENSE)。软件按「现状」提供，不附带任何明示或默示担保，
包括但不限于适销性、特定用途适用性和不侵权担保。作者不对使用本软件造成的任何直接或间接损失负责，
包括但不限于数据丢失、设备损坏、账号受限、法律纠纷。

**二、与第三方无关**

本项目与哔哩哔哩、GitHub、SenseVoice、PaddleOCR、Silero 等任何公司或项目**没有隶属、合作或背书关系**。
文中出现的商标、产品名归各自所有者。

**三、内容版权**

通过本工具下载或解析的视频、音频、字幕、封面、评论、弹幕，**版权归原作者及原平台所有**。
本工具只做格式转换与本地处理，不主张任何权利。请：

- 仅用于个人学习、研究、无障碍辅助等合理使用场景；
- 不要公开传播、二次上传、批量抓取或用于商业用途；
- 遵守目标网站的服务条款、robots 协议与所在地区的著作权法。

因使用本工具产生的版权纠纷，由使用者自行承担。

**四、账号与凭证**

`cookies.txt` 等同账号密码，可读取你的私信、投币、收藏等信息。请自行妥善保管，
不要提交到任何仓库、不要发给他人。本仓库已在 `.gitignore` 中排除该文件，
但**历史提交无法自动追溯**——若你曾误提交，请立即到站点退出全部登录以作废凭证。

**五、识别结果**

OCR 与语音转写的输出**可能有错**，尤其在唱腔、方言、强背景音乐、低分辨率画面上。
结果仅供辅助理解，不应作为证据、字幕成品或正式记录使用。

**六、模型许可**

各模型遵循其上游许可，使用前请自行查阅：

- PP-OCRv4（PaddleOCR）：Apache-2.0
- SenseVoice：见 [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- Silero VAD：MIT
- sherpa-onnx：Apache-2.0

模型文件不随本仓库分发。
