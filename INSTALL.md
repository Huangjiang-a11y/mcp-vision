# 安装指南

`mcp-vision` 是一个 MCP 服务器：把视频变成 AI 能读的东西（帧、网格页、音频电平、基频、字幕、转写）。
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
| 内存 | ≥ 1 GB 空闲 | 转写模型加载后常驻约 340 MB（按需加载，不转写不占） |

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

## 3. 模型与环境变量

模型放哪、叫什么名字都不重要，**用环境变量指过去就行**。

### 3.1 OCR（画面文字，约 16 MB）

```sh
mkdir -p ~/models
HF="https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4"
curl -L -o ~/models/det.onnx  "$HF/ch_PP-OCRv4_det_infer.onnx"
curl -L -o ~/models/rec.onnx  "$HF/ch_PP-OCRv4_rec_infer.onnx"
curl -L -o ~/models/keys.txt \
  "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt"

export MVV_OCR_DET=~/models/det.onnx
export MVV_OCR_REC=~/models/rec.onnx
export MVV_OCR_KEYS=~/models/keys.txt
```

魔搭的转存（同一份文件，字节数一致）：

```sh
MS="https://www.modelscope.cn/models/deadash/paddleocr/resolve/master"
curl -L -o ~/models/det.onnx "$MS/ch_PP-OCRv4_det_infer.onnx"
curl -L -o ~/models/rec.onnx "$MS/ch_PP-OCRv4_rec_infer.onnx"
curl -L -o ~/models/keys.txt "$MS/ppocr_keys_v1.txt"
```

三个变量里缺哪个，那个就回落到仓库 `models/` 下的默认文件名，所以只设一部分也能跑。

### 3.2 转写（语音 → 文字，约 229 MB）

两个模型：断句用 **silero VAD**，识别用 **SenseVoice**（来自 FunAudioLLM / FunASR，© 阿里巴巴集团）。

```sh
MS="https://www.modelscope.cn/models/xiaowangge/sherpa-onnx-sense-voice-small/resolve/master"

curl -L -o ~/models/asr.onnx   "$MS/model_q8.onnx"
curl -L -o ~/models/asr.tokens "$MS/tokens.txt"
curl -L -o ~/models/vad.onnx \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

export MVV_ASR_MODEL=~/models/asr.onnx
export MVV_ASR_TOKENS=~/models/asr.tokens
export MVV_VAD_MODEL=~/models/vad.onnx
```

三个都是**必填**，少一个 `transcribe` 会直接报缺哪个变量。

要点：

- **VAD 别用魔搭同仓库里的 `silero-vad/model.onnx`（2.3 MB）**：那是另一版导出，喂给
  sherpa-onnx 的 `VoiceActivityDetector` 会一段都切不出来（实测 30 秒人声 0 段）。
- 上面那份 `model_q8.onnx` 与 sherpa-onnx 官方 int8 对同一段音频输出逐字相同，
  `tokens.txt` 的 md5 也一致（`17cdd9d5…`）。
- 官方 1 GB 压缩包（含 fp32 + int8 两套）只在上面不通时用，解开后把 `model.int8.onnx`
  和 `tokens.txt` 指给前两个变量即可。
- `MVV_ASR_MODEL` 建议用 int8 权重（229 MB）；fp32 是 894 MB，慢且占内存。

### 3.3 变量总表

| 变量 | 用途 | 必填 |
|---|---|---|
| `MVV_OCR_DET` | PP-OCRv4 检测模型 | OCR 时必填 |
| `MVV_OCR_REC` | PP-OCRv4 识别模型 | OCR 时必填 |
| `MVV_OCR_KEYS` | 字典 `ppocr_keys_v1.txt` | OCR 时必填 |
| `MVV_ASR_MODEL` | SenseVoice onnx | 转写时必填 |
| `MVV_ASR_TOKENS` | SenseVoice 的 tokens | 转写时必填 |
| `MVV_VAD_MODEL` | silero VAD onnx | 转写时必填 |
| `MVV_ORT_THREADS` | ORT 线程数，默认 2 | 否 |
| `MVV_ORT_PROVIDERS` | ORT 加速器，默认 cpu | 否 |

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
      "args": ["/absolute/path/to/mcp-vision/mcp_server.py"],
      "env": {
        "MVV_OCR_DET": "/home/me/models/det.onnx",
        "MVV_OCR_REC": "/home/me/models/rec.onnx",
        "MVV_OCR_KEYS": "/home/me/models/keys.txt",
        "MVV_ASR_MODEL": "/home/me/models/asr.onnx",
        "MVV_ASR_TOKENS": "/home/me/models/asr.tokens",
        "MVV_VAD_MODEL": "/home/me/models/vad.onnx"
      }
    }
  }
}
```

两点：

- 必须用**绝对路径**（脚本和模型都是），客户端的工作目录不一定是仓库目录。
- **`env` 块里的变量不会自动继承 shell 的 `export`**，需要显式写进去。只做抽帧/电平分析的话整块可以省略。

## 5. 验证

```sh
sh check.sh
```

期望输出（版本号随机器不同）：

```
mcp-vision 环境体检
  [OK] python3
  [OK] ffmpeg
  [OK] ffprobe
  [OK] numpy <版本>
  [OK] onnxruntime <版本>
  [OK] yt_dlp
  [OK] sherpa_onnx
  [OK] STT 模型（MVV_ASR_MODEL / MVV_ASR_TOKENS / MVV_VAD_MODEL）
  [OK] OCR 模型
冒烟：MCP tools/list
  工具: ['workspace', 'video', 'audio', 'subtitles']
  action: ['open', 'list']
```

`[--]` 表示少了个可选能力，其余照样能跑；`[!!]` 表示命令本身不在 PATH 上。

冒烟测试（带 `--test` 会额外跑不需要模型和素材的回归测试）：

```sh
sh check.sh --test
```

手动试一遍四个工具：

```
workspace open  "file:///path/to/a.mp4"
video     quantity=15
audio     view=pitch time=0
audio     view=history time=0 seconds=15 [frame_ms=5~200]
audio     view=spectrogram time=0 seconds=15
subtitles mode=stt time=0 seconds=30
```

## 6. 目录约定

```
mcp-vision/
├── models/            OCR 权重默认位置（gitignore，可被环境变量覆盖）
├── tests/             不需要模型/素材的回归测试
├── scripts/           基准脚本（需要模型或运行中的服务）
├── workspace/         产物：每个视频一个文件夹（gitignore）
│   ├── latest.txt     最近打开的媒体路径
│   └── <标题>/
│       ├── meta.json  id / 标题 / 时长 / 媒体路径
│       ├── <原文件名>  媒体副本
│       ├── video/     t<毫秒>-<宽>x<高>.jpg 帧、sheet_<MMSS>_<列>x<行>_<hash>.png/.jpg 网格页、.scene-<hash>.json 扫描缓存
│       ├── audio/     s-<起>-<止>.pcm（分析临时文件）、a-<起>-<止>-<view>.png
│       └── subtitles/ 字幕段直接写在返回里，不落盘
├── cookies.txt        B 站登录态（gitignore，**绝不要提交**）
└── out/               实验产物（gitignore）
```

## 7. 常见问题

**`sherpa_onnx` 装不上**
Termux 上 `pkg install python-onnxruntime` 后再 `pip install sherpa-onnx`；如果 pip 编译失败，说明该平台没有预编译 wheel，改用别的设备跑转写。

**转写返回空**
先确认视频有音轨：`ffprobe -show_streams`。纯 BGM 无对白的片子，VAD 会判成「没有语音」，属正常。

**下载中断 / 太慢**
`curl -C -` 可以续传：`curl -C - -L -o 目标文件 "地址"`。下完用 `ls -l` 对一下字节数。

**转写结果错字多**
唱腔、强 BGM、方言都会掉准确率。转写没有 `language` / `max_speech` 这类调参口径：语种由 SenseVoice 自己判（结果里的 `lang`），断句由 VAD 自动切。错字多时换更干净的窗口（`time` / `seconds`）重试，或用 `mode=ocr` 核对画面原文。

**OCR 报 `No module named onnxruntime`**
装 `python-onnxruntime`（Termux）或 `pip install onnxruntime`。

**下载 B 站视频 403 / 需要登录**
把浏览器 cookies 导出成 Netscape 格式存为 `cookies.txt`。该文件含账号凭证，别分享、别提交。

**抽帧很慢**
`detail` 默认已是 `low`；再想快就调小 `quantity`、用 `crop` 缩小取景，或者接受解码本身受手机 CPU 限制。

---

## 免责声明

**零、项目定位**

本项目**仅供学习、研究使用**。

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
- 不要公开传播、二次上传、批量抓取；**本工具不授予任何内容版权**；
- 遵守目标网站的服务条款、robots 协议与所在地区的著作权法。

因使用本工具产生的版权纠纷，由使用者自行承担。

**四、账号与凭证**

`cookies.txt` 等同账号密码，可读取你的私信、投币、收藏等信息。请自行妥善保管，
不要提交到任何仓库、不要发给他人。本仓库已在 `.gitignore` 中排除该文件，
但**历史提交无法自动追溯**——若你曾误提交，请立即到站点退出全部登录以作废凭证。

**五、识别结果**

OCR 与语音转写的输出**可能有错**，尤其在唱腔、方言、强背景音乐、低分辨率画面上。
结果仅供辅助理解，不应作为证据、字幕成品或正式记录使用。

**六、模型与依赖许可**

各模型遵循其上游许可，使用前请自行查阅：

- PP-OCRv4（PaddleOCR）：Apache-2.0
- SenseVoice（FunAudioLLM / FunASR，© 阿里巴巴集团）：**代码 MIT，权重另算**——
  权重遵循 FunASR Model Open Source License v1.1（模型卡标 `license: other`），
  **须保留模型名 SenseVoice 并署名来源**。注意它不是 Apache-2.0。
- Silero VAD：MIT
- sherpa-onnx：Apache-2.0

ffmpeg / ffprobe 是以**子进程**调用的外部程序，本仓库不链接、不捆绑；
若你要在自己的分发物里附带 ffmpeg 二进制，需遵守其 LGPL/GPL 条款。
完整的第三方许可清单见 [THIRD_PARTY.md](THIRD_PARTY.md)。

**七、模型分发**

本仓库**不包含任何模型权重**，也不附带下载脚本。INSTALL.md 里只给出去往第三方托管地的地址，
文件由你自行获取；这些地址的可用性、访问条款与许可由托管方决定，与本项目无关。
