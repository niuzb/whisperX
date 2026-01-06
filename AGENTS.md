<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# WhisperX 代码库架构总览 (AGENTS 指南)

> 本文面向需要在本仓库上进行开发、重构、测试或调试的代理 (human/AI)，帮助快速建立对整体架构和关键约定的认识。

---

## 1. 项目概览

### 1.1 项目目标

WhisperX 是在 OpenAI Whisper 基础上的一个增强型语音识别 (ASR) 与处理流水线，核心特性包括：

- 使用 Faster-Whisper + CTranslate2 提供 **高速批量推理**；
- 使用 wav2vec2 / torchaudio / HF 模型做 **强制对齐 (forced alignment)**，得到 **词级时间戳**；
- 基于 pyannote-audio (或 silero) 做 **VAD 语音活动检测** 与 **多说话人分离 (speaker diarization)**；
- 提供命令行工具 `whisperx`、Python API 以及 **HTTP API Server**，支持批量任务提交与查询。

整体定位是：在尽量复用 Whisper 原有行为的前提下，提供更精细的时间对齐、更高吞吐量和多说话人支持。

### 1.2 模块划分与调用关系

核心包位于 `whisperx/` 目录下，各模块大致职责如下：

- **入口与公共接口**
  - `whisperx/__main__.py:cli`：命令行入口，解析 CLI 参数并调用 `transcribe_task`。
  - `whisperx/__init__.py`：对外暴露高层 API (`load_model`, `align`, `diarize` 等)。
  - `whisperx/api_server.py`：**[新增]** 基于 FastAPI 的 HTTP 服务入口，提供异步任务提交 (`/submit`) 与状态查询 (`/query`) 接口。

- **ASR 与 VAD**
  - `whisperx/asr.py`：封装 Faster-Whisper + VAD。
    - `FasterWhisperPipeline`：集成 VAD 切分、批量推理、语言检测。
    - `load_model(...)`：加载 ASR 模型的统一入口。

- **对齐 (Alignment)**
  - `whisperx/alignment.py`：使用 wav2vec2/torchaudio 模型进行强制对齐，输出字符/词级时间戳。

- **说话人分离 (Diarization)**
  - `whisperx/diarize.py`：封装 pyannote pipeline，输出说话人标签并分配给对应的词。

- **转录任务调度**
  - `whisperx/transcribe.py`：CLI 调度核心，串联 ASR -> 对齐 -> Diarization -> 输出。
  - `whisperx/task_results/`：**[新增]** API Server 用于存储任务状态与结果的本地目录 (JSON 格式)。

- **工具与 Writer**
  - `whisperx/utils.py`：格式转换、文件输出 (`ResultWriter` 及其子类)。
  - `whisperx/log_utils.py`：统一日志配置。

---

## 2. 构建与命令 (Build & Commands)

### 2.1 安装与依赖管理

依赖在 `pyproject.toml` 中配置，推荐使用 `uv` 管理。

- **核心依赖**：`ctranslate2`, `faster-whisper`, `torch`, `torchaudio`, `pyannote-audio`。
- **服务端依赖**：`fastapi`, `uvicorn`, `pydantic` (注意：当前 `pyproject.toml` 未包含这些依赖，运行 server 前需手动安装)。

开发环境安装：
```bash
uv sync --all-extras --dev
# 或手动安装服务端依赖
pip install fastapi uvicorn pydantic
```

### 2.2 运行方式

#### 命令行工具 (CLI)
```bash
# 基本用法
whisperx audio.wav --model large-v2 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H

# 开启 Diarization
whisperx audio.wav --model large-v2 --diarize --highlight_words True
```

#### API 服务 (Server)
提供了基于 HTTP 的异步任务处理能力：

```bash
# 启动开发服务器
sh start.sh
# 或
uvicorn whisperx.api_server:app --reload --host 0.0.0.0 --port 8000
```

- **接口说明**：
  - `POST /submit`: 提交转录任务，返回 `task_id`。
  - `POST /query`: 通过 Header 中的 `X-Api-Request-Id` 查询任务状态与结果。
  - 均支持自定义 Header 如 `X-Tt-Logid` 用于链路追踪。

---

## 3. 代码风格与约定 (Code Style)

- **模块化设计**：功能模块 (ASR, Align, Diarize) 解耦，通过 `transcribe.py` 或 `api_server.py` 组装。
- **API 交互规范**：
  - HTTP API 使用自定义 Header (`X-Api-Status-Code`, `X-Api-Message`) 传递业务状态，Body 返回业务数据。
  - 任务状态码定义在 `whisperx.api_server.TaskStatus` 类中 (如 `20000000` 表示成功)。
- **类型提示**：广泛使用 `typing` 模块与 `pydantic` 模型 (针对 API 请求体)。
- **日志**：统一使用 `whisperx.log_utils` 获取 logger。

---

## 4. 测试 (Testing)

目前仓库未集成统一测试框架 (pytest/unittest)。

- **CLI 验证**：使用 README 中的示例命令对 sample 音频进行测试。
- **API 验证**：启动 server 后，使用 curl 或 Postman 调用 `/submit` 和 `/query` 接口验证流程。
- **开发建议**：新增功能时建议编写简单的 Python 脚本或 curl 命令序列进行回归测试。

---

## 5. 安全与隐私 (Security)

1. **模型来源**：依赖 Hugging Face 与 PyTorch 官方源，确保模型文件可信。
2. **API 安全**：
   - 当前 API Server **未实现强身份验证**，Header 中的 `X-Api-App-Key` 等仅作透传，生产部署需在网关层处理鉴权。
   - 任务结果以 JSON 文件形式存储在 `whisperx/task_results/`，无自动清理机制，需注意磁盘占用与敏感数据生命周期管理。
3. **Token 管理**：Diarization 依赖 HF Token，建议通过环境变量传递，避免硬编码。
4. **资源限制**：Batch size 和并发数需在调用侧控制，防止 GPU 显存耗尽。

---

## 6. 配置与环境 (Configuration)

- **环境要求**：Python >= 3.9，推荐 CUDA 12.8 (Linux/Windows)。
- **API 配置**：
  - 端口默认 `8000`，绑定 `0.0.0.0` (如使用命令行参数)。
  - 任务存储路径默认为 `whisperx/task_results/`。
- **推理配置**：
  - CLI 通过参数 (`--model`, `--device`, `--compute_type`) 控制。
  - API Server 目前主要演示了任务提交/查询框架，具体推理参数需在 `submit_task` 逻辑中完善 (当前为 mock 实现)。
