# 第三方许可

本仓库（`mcp-vision`）的代码以 MIT 发布，见 [LICENSE](LICENSE)。
**LICENSE 只覆盖本仓库里的代码**，不覆盖下面这些：

- 模型权重（本仓库不含，由使用者自行获取）
- 经由本工具下载或解析的任何音视频、字幕、封面、弹幕、评论
- 运行时的第三方库与外部程序

## 运行时依赖

| 组件 | 许可 | 在本项目中的角色 |
|---|---|---|
| numpy | BSD-3-Clause¹ | 数组与 FFT，核心依赖 |
| onnxruntime | MIT | 跑 OCR 模型（可选） |
| sherpa-onnx | Apache-2.0 | 跑 SenseVoice / Silero VAD（可选） |
| yt-dlp | Unlicense（公有领域） | B 站等链接下载（可选） |
| ffmpeg / ffprobe | LGPL-2.1+ 或 GPL-2.0+ | **外部程序**，见下 |

¹ GitHub 把它标成 `NOASSERTION`：条款本身三条俱全，只是把「版权人姓名」换成了
「NumPy Developers」，自动识别匹配不上标准模板。按 BSD-3-Clause 对待即可。

以上都是运行时被调用，没有静态链接，也没有把它们的源码并入本仓库。
按 MIT 分发本仓库不需要额外义务；你重新分发它们时需遵守各自条款。

## ffmpeg 特别说明

本项目通过 `subprocess` 调用 `ffmpeg` / `ffprobe` 命令行，**不链接、不捆绑、不修改**。
这种「调用外部程序」的用法不构成本项目的衍生作品，所以本仓库可以是 MIT。

但如果你要在自己的分发物里**附带 ffmpeg 二进制**，就是另一回事：

- 只带 LGPL 构建（`--disable-gpl`）：需提供源码或获取途径，并允许替换库
- 带了 GPL 组件（如 `libx264`、`--enable-gpl`）：整个 ffmpeg 包按 GPL 分发
- Termux / 各家包管理器装的通常是 GPL 构建，随包分发时请核对

## 模型权重

**本仓库不包含任何模型权重，也没有下载脚本。** [INSTALL.md](INSTALL.md) 只给出托管地址，
文件由你自行获取，许可由你与模型提供方之间确定：

| 模型 | 许可 | 备注 |
|---|---|---|
| PP-OCRv4（PaddleOCR） | Apache-2.0 | 需保留版权与许可声明 |
| SenseVoiceSmall | FunASR Model Open Source License v1.1 | 见下节，**不是 Apache** |
| Silero VAD | MIT | — |

### SenseVoice 是两套许可，别搞混

| 层面 | 许可 | 出处 |
|---|---|---|
| **代码**（推理库、训练脚本） | **MIT** | `FunAudioLLM/SenseVoice`、`QwenAudio/SenseVoice`、`modelscope/FunASR` 三个仓库均为 MIT |
| **模型权重**（`model.pt` / 转出的 onnx） | **FunASR Model Open Source License v1.1** | 模型卡写 `license: other`，`license_link` 指向 [`modelscope/FunASR/MODEL_LICENSE`](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE) |

容易记混的地方：

- **Apache-2.0 不是 SenseVoice**。本项目里 Apache-2.0 有两个：`sherpa-onnx`（推理框架）和
  PP-OCRv4（PaddleOCR 的模型）。SenseVoice 两者都不是。
- **MIT 只覆盖代码**。你实际下载的权重走 FunASR 那份协议，署名义务来自它而不是 MIT。

协议 v1.1 的实际条款（已核对原文）只有三类：§2.1 允许使用/复制/修改/分享；
§2.2 要求**注明出处与作者、保留模型名**；§3–§7 是免责、社区行为与终止条款。

两个值得留意的细节：

1. **§3 写着「仅作为参考和学习使用」**。本项目也仅供学习研究，
   两者一致，不涉及超出该范围的用途。
   若你要把权重用到别处，请自行向上游确认许可范围并留档。
2. **§4.2 是条不常见的约束**：不得对模型作「无正当理由的贬低、恶意抹黑」，
   违反即自动失去全部许可。这是评述性条款，正常使用不会碰到，但要知道它存在。

本项目对 §2.2 的落地方式：文档中始终以「SenseVoice」这一名称指代该模型，
并在 [README](README.md) 与 [INSTALL](INSTALL.md) 中注明来自 FunAudioLLM / FunASR / 阿里巴巴集团。

## 思路借鉴

[README](README.md) 的「致谢」列出了三个参考项目，三者均为 **MIT**：

- yangjili6/video-understanding-kit
- Tests4gptplay/dsh-video-vision
- ppppiii/dsh-bili-watch

借鉴的是**分层与接口设计**（`workspace` / `video` / `audio` / `subtitles` 四层、
按时间点取原帧、链接规范化），属于思想与方法，未复制其源码；运行时也不依赖它们。
因此没有触发 MIT 的「保留版权声明」义务——不过致谢仍然保留，这是应该做的。

## 关于点阵字模

`audiochart.py` 里的 5×7 点阵字形（47 个字符，仅用于在图上写标题与刻度）是功能性的位图数据。
这类点阵字体在多数法域不被视为受版权保护的美术作品（美国明确不保护字体字形本身），
风险很低；若需要绝对干净，可自行重绘或用 matplotlib 等绘图库替代。

## 方法参考：phyphox（只参考文档，不复用代码）

`engine.py` 里基频自相关的方法（100ms 分析块、"超过最大幅度一半"的峰门限、
"最后一个极大 ÷ 极大数量"求周期）参考了 phyphox 的
[Audio Autocorrelation](https://phyphox.org/wiki/index.php?title=Experiment:_Audio_Autocorrelation)、
[Frequency History](https://phyphox.org/wiki/index.php?title=Experiment:_Frequency_History)、
[Audio Amplitude](https://phyphox.org/wiki/index.php?title=Experiment:_Audio_Amplitude)
三篇**实验文档**。

**没有复用 phyphox 的任何源码。** [phyphox-android](https://github.com/phyphox/phyphox-android)
以 **GPL-3.0** 发布，本项目是 MIT；因此这里只依据公开文档描述的方法独立实现，
不拷贝、不翻译其源代码，避免许可冲突。方法、算法本身不受版权保护。

`phyphox` 与 `RWTH Aachen University` 是 RWTH Aachen University 的注册商标，
本项目与之**没有隶属、合作或背书关系**。

## 内容版权

通过本工具处理的一切视频、音频、字幕、封面、评论、弹幕，版权归原作者与平台所有。
本项目只做本地解码与格式转换，不主张任何权利。详见 [INSTALL.md](INSTALL.md) 的「免责声明」。
