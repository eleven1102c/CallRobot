# CallRobot GPU Server

这个目录搭好了 GPU 服务器侧的全双工对话模块骨架：

- Streaming ASR: FunASR online Paraformer
- Dialogue LLM: Qwen2.5-7B-Instruct via vLLM，支持 streaming tokens 和 `abort(request_id)` 取消
- Interrupt-VAD: 结合对话状态、正在播报内容、ASR partial text 判断真/假打断
- Dialogue State Manager: `LISTENING`、`THINKING`、`BOT_SPEAKING`、`USER_INTERRUPTING`
- TTS Streaming: CosyVoice，按文本片段输出音频 chunk
- WebSocket API: Mac 本地音频采集/Fast VAD/播报模块通过 `/ws` 对接

## 部署

GPU 服务器需要 NVIDIA 驱动、Docker、NVIDIA Container Toolkit。

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm callrobot-gpu python3.10 gpu_server/scripts/download_models.py
docker compose -f docker-compose.gpu.yml up
```

健康检查：

```bash
curl http://GPU_SERVER_IP:9000/health
```

## NVIDIA 4090 24G 服务器部署

4090 24G 建议优先用裸机 Python venv 部署，调试 CosyVoice、CUDA、vLLM 更直接。Docker 也支持，但需要先把官方 CosyVoice 仓库准备到 `third_party/CosyVoice`。

### 1. 服务器基础检查

在 4090 服务器上确认驱动和 CUDA 可用：

```bash
nvidia-smi
```

建议环境：

- Ubuntu 22.04/24.04
- NVIDIA Driver 支持 CUDA 12.x
- Python 3.10 或 3.11。Python 3.12 依赖兼容性更容易出问题。

安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg sox libsndfile1 python3-venv python3-dev build-essential
```

### 2. 拉取项目

```bash
git clone YOUR_REPO_URL CallRobot
cd CallRobot
```

如果你还没有上传 GitHub，也可以把项目目录 scp 到服务器。

### 3. 准备 Python 环境

```bash
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
bash gpu_server/scripts/install_4090.sh
```

`install_4090.sh` 会安装 GPU 侧依赖，并把官方 CosyVoice clone 到：

```text
third_party/CosyVoice
```

### 4. 准备模型

默认模型目录：

```text
./models
```

如果服务器网络可用：

```bash
MODEL_DIR=./models python gpu_server/scripts/download_models.py
```

如果你已经有模型文件，建议目录结构保持：

```text
models/
  Qwen2.5-7B-Instruct/
  speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/
  speech_fsmn_vad_zh-cn-16k-common-pytorch/
  punc_ct-transformer_zh-cn-common-vocab272727-pytorch/
  CosyVoice-300M-SFT/
```

如果使用 7B-AWQ：

```bash
export QWEN_MODEL=/path/to/Qwen2.5-7B-Instruct-AWQ
export QWEN_QUANTIZATION=awq
```

4090 24G 通常可以优先尝试 FP16 的 Qwen2.5-7B-Instruct：

```bash
export QWEN_MODEL=$PWD/models/Qwen2.5-7B-Instruct
export QWEN_QUANTIZATION=
```

### 5. 启动服务

默认监听 `0.0.0.0:9000`：

```bash
source .venv-gpu/bin/activate
bash gpu_server/scripts/start_4090.sh
```

4090 启动脚本默认参数：

```bash
QWEN_GPU_MEMORY_UTILIZATION=0.82
QWEN_MAX_MODEL_LEN=8192
QWEN_MAX_TOKENS=512
QWEN_MAX_NUM_SEQS=2
QWEN_MAX_NUM_BATCHED_TOKENS=4096
QWEN_ENFORCE_EAGER=false
FUNASR_USE_VAD=false
```

如果启动 OOM，先降上下文：

```bash
export QWEN_MAX_MODEL_LEN=4096
export QWEN_MAX_NUM_BATCHED_TOKENS=2048
export QWEN_GPU_MEMORY_UTILIZATION=0.76
bash gpu_server/scripts/start_4090.sh
```

健康检查：

```bash
curl http://SERVER_IP:9000/health
```

如果外网访问不到，检查云服务器安全组和系统防火墙是否放行 TCP `9000`。

### 6. Mac 客户端连接 4090 服务器

文字模式：

```bash
./mac_client/scripts/run_mac_client.sh \
  --server ws://SERVER_IP:9000/ws \
  --no-mic
```

全双工语音：

```bash
./mac_client/scripts/run_mac_client.sh \
  --server ws://SERVER_IP:9000/ws \
  --input-device 0 \
  --output-device 1 \
  --debug-audio
```

### 7. Docker 部署方式

先准备 CosyVoice 官方仓库：

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
```

构建并启动：

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml up
```

Docker 方式会挂载：

```text
./models -> /models
./third_party/CosyVoice -> /app/third_party/CosyVoice
```

## Kaggle Notebook GPU 部署

你的 GPU 服务器如果是 Kaggle Notebook 容器，并且已经通过 frp 把容器内 `8081` 端口暴露到公网，不要用 Docker 部署。直接在 notebook 里安装依赖并启动 `uvicorn` 监听 `0.0.0.0:8081`。

### 1. 准备 Notebook

Kaggle Notebook 右侧设置：

- Accelerator: 选择 GPU。
- Internet: 打开，否则无法 `pip install` 或下载模型。
- Persistence: 建议打开，避免每次重启都重新下载模型。

在 notebook 里先确认 GPU：

```bash
!nvidia-smi
```

### 2. 上传或拉取项目代码

方式 A：把本项目打包上传到 Kaggle，然后解压到 `/kaggle/working/CallRobot`。

方式 B：如果你把项目放到了 Git 仓库：

```bash
%cd /kaggle/working
!git clone YOUR_REPO_URL CallRobot
%cd /kaggle/working/CallRobot
```

后续命令都假设项目目录是 `/kaggle/working/CallRobot`。

### 3. 安装依赖

```bash
%cd /kaggle/working/CallRobot
!bash gpu_server/scripts/kaggle_install.sh
```

不要用 `pip install cosyvoice` 作为 CosyVoice 安装方式。这个包名在 pip 上不一定对应官方源码结构，常见现象就是：

```text
ModuleNotFoundError: No module named 'cosyvoice.cli'
```

正确方式是在 Kaggle 里 clone 官方 CosyVoice 仓库，并安装它自己的依赖：

```bash
%cd /kaggle/working
!git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
%cd /kaggle/working/CosyVoice
!pip install -r requirements.txt
```

然后设置官方仓库路径：

```bash
%env COSYVOICE_REPO_DIR=/kaggle/working/CosyVoice
```

可以先用下面方式确认其他核心依赖是否安装成功：

```bash
!python -c "import fastapi, uvicorn, funasr, vllm, transformers; print('core deps ok')"
```

CosyVoice 的验证方式：

```bash
!PYTHONPATH=/kaggle/working/CosyVoice:/kaggle/working/CosyVoice/third_party/Matcha-TTS python -c "from cosyvoice.cli.cosyvoice import CosyVoice; print('cosyvoice ok')"
```

### 4. 准备模型

默认模型目录是 `/kaggle/working/models`。如果 notebook 网络可用，可以直接下载：

```bash
%cd /kaggle/working/CallRobot
!MODEL_DIR=/kaggle/working/models python gpu_server/scripts/download_models.py
```

如果你已经把模型作为 Kaggle Dataset 挂载，建议不要复制大模型，直接把环境变量指向 `/kaggle/input/...` 下的实际目录。例如：

```bash
%env MODEL_DIR=/kaggle/input/callrobot-models
%env QWEN_MODEL=/kaggle/input/callrobot-models/Qwen2.5-7B-Instruct
%env FUNASR_MODEL=/kaggle/input/callrobot-models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
%env FUNASR_USE_VAD=false
%env FUNASR_VAD_MODEL=/kaggle/input/callrobot-models/speech_fsmn_vad_zh-cn-16k-common-pytorch
%env FUNASR_PUNC_MODEL=/kaggle/input/callrobot-models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
%env COSYVOICE_MODEL=/kaggle/input/callrobot-models/CosyVoice-300M-SFT
%env COSYVOICE_REPO_DIR=/kaggle/working/CosyVoice
```

`CosyVoice-300M-SFT` 是 CosyVoice 1 模型，目录里应当有 `cosyvoice.yaml`。`CosyVoice2-*` 模型目录里才会有 `cosyvoice2.yaml`。服务会根据模型目录里的 yaml 自动选择 `CosyVoice` 或 `CosyVoice2`。

当前架构由 Mac 端做 Fast VAD 和 endpointing，所以 GPU 侧 FunASR 默认只做 streaming ASR，不再加载 FunASR VAD。不要把 `FUNASR_USE_VAD` 打开，否则 FunASR VAD 可能收到 ASR 的 `chunk_size=[5,10,5]` 并触发类型错误。

如果日志里仍然出现 `/kaggle/working/models/CosyVoice-300M-SFT/... not found`，说明 `COSYVOICE_MODEL` 没有被设置到你的 Dataset 路径，启动前重新执行：

```bash
%env COSYVOICE_MODEL=/kaggle/input/你的dataset目录/CosyVoice-300M-SFT
```

Kaggle 免费 GPU 显存通常比较紧。Qwen2.5-7B + ASR + TTS 同进程跑不动时，先调低 vLLM 显存占用：

```bash
%env QWEN_GPU_MEMORY_UTILIZATION=0.72
%env QWEN_MAX_MODEL_LEN=4096
%env QWEN_MAX_TOKENS=256
```

### 5. 启动 GPU 服务到 8081

在 notebook cell 中运行：

```bash
%cd /kaggle/working/CallRobot
!GPU_SERVER_PORT=8081 bash gpu_server/scripts/kaggle_start.sh
```

这个 cell 会一直占用运行，看到 `Uvicorn running on http://0.0.0.0:8081` 后，说明容器内服务启动了。

因为你已经用 frp 暴露了容器 `8081`，公网访问地址应当是：

```text
http://你的-frp-公网域名或IP/health
ws://你的-frp-公网域名或IP/ws
```

如果 frp 公网侧不是默认 HTTP 端口，例如映射到了 `18081`：

```text
http://你的-frp-公网域名或IP:18081/health
ws://你的-frp-公网域名或IP:18081/ws
```

### 6. 从 Mac 端连接 Kaggle GPU 服务

先测健康检查：

```bash
curl http://你的-frp-公网域名或IP:公网端口/health
```

再用 Mac 客户端文字模式测试：

```bash
./mac_client/scripts/run_mac_client.sh \
  --server ws://你的-frp-公网域名或IP:公网端口/ws \
  --no-mic
```

最后做全双工麦克风测试：

```bash
./mac_client/scripts/run_mac_client.sh \
  --server ws://你的-frp-公网域名或IP:公网端口/ws \
  --input-device 0 \
  --output-device 1
```

### 7. Kaggle 部署注意事项

- Kaggle Notebook 重启后，`/kaggle/working` 可能保留但运行进程不会保留，需要重新启动 `kaggle_start.sh`。
- frp 必须在 GPU 服务启动前或启动后保持运行，并确认转发目标是容器内 `127.0.0.1:8081` 或 `0.0.0.0:8081`。
- 如果 `/health` 能访问但 `/ws` 连不上，优先检查 frp 是否支持 WebSocket upgrade，以及公网 URL 是否用了正确的 `ws://` 或 `wss://`。
- 如果模型加载 OOM，先降低 `QWEN_GPU_MEMORY_UTILIZATION`、`QWEN_MAX_MODEL_LEN`，或临时换更小的 Qwen 模型跑通链路。

### 8. Qwen OOM 调整

Kaggle 常见 T4/P100 只有 16GB 显存。原始 FP16 的 Qwen2.5-7B-Instruct 权重大约已经接近显存上限，再加 vLLM KV cache、ASR、TTS，很容易启动 OOM。当前 `kaggle_start.sh` 已经使用较保守默认值：

```bash
QWEN_GPU_MEMORY_UTILIZATION=0.72
QWEN_MAX_MODEL_LEN=4096
QWEN_MAX_TOKENS=256
QWEN_MAX_NUM_SEQS=1
QWEN_MAX_NUM_BATCHED_TOKENS=2048
QWEN_ENFORCE_EAGER=true
```

如果还是 OOM，按这个顺序调：

1. 先把上下文降到 2048：

```bash
%env QWEN_MAX_MODEL_LEN=2048
%env QWEN_MAX_NUM_BATCHED_TOKENS=1024
%env QWEN_MAX_TOKENS=128
%env QWEN_GPU_MEMORY_UTILIZATION=0.65
```

2. 如果权重本身就放不下，换 Qwen2.5-7B 的 AWQ/GPTQ 量化模型，并设置：

```bash
%env QWEN_MODEL=/kaggle/input/callrobot-models/Qwen2.5-7B-Instruct-AWQ
%env QWEN_QUANTIZATION=awq
%env QWEN_DTYPE=float16
```

3. 如果仍然差一点显存，可以临时打开 CPU offload，代价是速度下降：

```bash
%env QWEN_CPU_OFFLOAD_GB=4
```

4. 如果只是为了先跑通链路，直接换小模型：

```bash
%env QWEN_MODEL=/kaggle/input/callrobot-models/Qwen2.5-3B-Instruct
%env QWEN_QUANTIZATION=
```

判断标准：

- OOM 出现在 `AsyncLLMEngine.from_engine_args` 附近：主要是 Qwen/vLLM 显存不够。
- OOM 出现在 ASR 或 TTS 加载后：先让 Qwen 独占更多显存，或者把 ASR/TTS 换小模型/拆服务。
- 16GB 显存上最稳方案是 `Qwen2.5-7B-Instruct-AWQ + max_model_len=2048/4096 + max_num_seqs=1`。

## Mac 测试端部署

Mac 端负责：

- 麦克风采集：16 kHz、单声道、PCM16。
- Fast VAD：本地 WebRTC VAD，过滤静音并做 endpointing。
- 轻量 AEC/噪音抑制：播放 TTS 时对低能量麦克风回声做门控，减少误触发。
- WebSocket：把 speech chunk 发送给 GPU `/ws`。
- 本地播报：接收 `tts_audio` WAV chunk 并排队播放。
- 打断处理：收到 `interrupt` 或 `cancelled` 立即停止当前播报并清空播放队列。

安装依赖：

```bash
python3 -m venv .venv-mac
source .venv-mac/bin/activate
pip install -r requirements-mac.txt
```

第一次使用麦克风时，macOS 会弹出权限授权。如果没有弹窗，到“系统设置 -> 隐私与安全性 -> 麦克风”里给 Terminal、iTerm 或你的 IDE 授权。

列出输入/输出设备：

```bash
./mac_client/scripts/run_mac_client.sh --list-devices
```

麦克风自检，确认 macOS 权限和输入设备是否真的采到声音：

```bash
./mac_client/scripts/check_mic.sh --input-device 0
```

正常说话时应看到 `rms` 和 `peak` 明显变化。如果一直接近 0，通常是麦克风权限、设备编号或输入源选错。

先用文字模式测试 GPU 服务、LLM、TTS 和本地播放：

```bash
./mac_client/scripts/run_mac_client.sh \
  --server ws://GPU_SERVER_IP:9000/ws \
  --no-mic
```

全双工麦克风测试：

```bash
./mac_client/scripts/run_mac_client.sh \
  --server ws://GPU_SERVER_IP:9000/ws \
  --input-device 0 \
  --output-device 1
```

如果全双工模式看起来“没反应”，打开音频调试：

```bash
./mac_client/scripts/run_mac_client.sh \
  --server ws://GPU_SERVER_IP:9000/ws \
  --input-device 0 \
  --output-device 1 \
  --debug-audio
```

Mac 客户端默认用 20ms 帧做 VAD，但会聚合成约 200ms 音频再发给 GPU，避免通过 frp 发送过多 WebSocket 小包。如果 `[user] speaking` 后长时间没有输出，可以先用更明确的 speaking 起止阈值测试：

```bash
./mac_client/scripts/run_mac_client.sh \
  --server ws://GPU_SERVER_IP:9000/ws \
  --input-device 0 \
  --output-device 1 \
  --debug-audio \
  --vad 3 \
  --speech-start-ms 120 \
  --speech-end-ms 500 \
  --max-utterance-ms 6000
```

本地 Mac 端维护 `user_speaking` 状态：

- 连续 6 帧有声音，默认 `20ms * 6 = 120ms`，进入 speaking，日志显示 `[user] speaking`。
- 连续 500ms 没有人声，退出 speaking，日志显示 `[user] stopped`，并发送 `end_utterance`。
- 可以通过 `--speech-start-ms` 和 `--speech-end-ms` 调整。

Mac 客户端默认启用轻量 AEC/噪音抑制。它不是专业自适应滤波器，而是在 VAD 前根据本地 TTS 播放状态、噪声底和麦克风能量做门控：

- 播放中且麦克风能量较低：认为是回声/背景声，置零后再送 VAD。
- 播放中但用户近场说话能量足够高：允许通过，用于打断。
- 安静时持续估计噪声底，抑制低能量环境声。

可调参数：

```bash
./mac_client/scripts/run_mac_client.sh \
  --server ws://GPU_SERVER_IP:9000/ws \
  --input-device 0 \
  --output-device 1 \
  --debug-audio \
  --aec-min-rms 220 \
  --aec-echo-rms 900
```

如果发现用户打断不灵敏，降低 `--aec-echo-rms`。如果机器人外放仍然误触发，提高 `--aec-echo-rms`。需要关闭时：

```bash
./mac_client/scripts/run_mac_client.sh --server ws://GPU_SERVER_IP:9000/ws --no-aec
```

排查判断：

- 没有 `[mic] rms=...`：麦克风流没启动，检查 macOS 麦克风权限、依赖安装、设备编号。
- 有 `[mic] rms=...` 但没有 `[user] speaking`：VAD 没触发，试试 `--vad 0` 或靠近麦克风说话。
- 有 `[user] speaking` 但没有 `[user] stopped`：环境一直被判定有声音，戴耳机、降低背景声，试 `--vad 3 --speech-end-ms 500`，或提高 `--aec-echo-rms`。
- 有 `[user] speaking` 但没有 `[mic] captured_frames=...`：WebSocket 发送可能被阻塞，确认 frp 链路稳定，或把 `--send-chunk-ms` 调到 `400`。
- 有 `[vad] end_utterance` 但没有 `[asr]`：音频已发出，检查 GPU 服务日志和 WebSocket 连接。

命令行里可以直接输入文本触发一轮对话，也可以使用：

- `/cancel`: 本地停止播放，并通知 GPU 取消当前 vLLM request 和 TTS。
- `/reset`: 重置当前会话。
- `/quit`: 退出客户端。

建议用耳机测试全双工打断，否则扬声器回放会被麦克风收进去，容易形成假打断或回声识别。若环境噪声大，可提高 VAD 激进度：

```bash
./mac_client/scripts/run_mac_client.sh --server ws://GPU_SERVER_IP:9000/ws --vad 3
```

默认模型路径在 `./models`，也可以在 `docker-compose.gpu.yml` 里改：

- `QWEN_MODEL=/models/Qwen2.5-7B-Instruct`
- `FUNASR_MODEL=/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online`
- `FUNASR_VAD_MODEL=/models/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- `FUNASR_PUNC_MODEL=/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`
- `COSYVOICE_MODEL=/models/CosyVoice-300M-SFT`

## WebSocket 协议

入口：`ws://GPU_SERVER_IP:9000/ws`

Mac 侧发送 JSON 文本消息：

```json
{
  "type": "audio",
  "session_id": "call-001",
  "audio_b64": "base64(pcm16 mono 16k chunk)"
}
```

常用 client event：

- `audio`: 发送 16k 单声道 PCM16 音频 chunk。Fast VAD 建议在 Mac 本地先过滤静音，只把 speech chunk 发到 GPU。
- `end_utterance`: 用户一句话结束；可带最后一段 `audio_b64`，也可只依赖最近 ASR partial。
- `text`: 直接用文本触发一轮 LLM+TTS，便于调试。
- `cancel`: 强制取消当前 LLM request 和 TTS。
- `reset`: 重置会话 ASR cache、历史和状态。

服务器返回 event：

- `state`: 状态变化。
- `asr_partial`: FunASR 流式中间文本。
- `asr_final`: 用户最终文本。
- `interrupt`: 判定为真打断，已经取消 LLM/TTS。
- `llm_token`: vLLM 流式 token delta。
- `tts_audio`: base64 WAV chunk，可直接给 Mac 播放队列。
- `bot_final`: 机器人完整回复。
- `cancelled`: 取消完成。

## 大致工作流

1. Mac 本地采集麦克风音频，做 resample 到 16k PCM16，Fast VAD 只保留 speech chunk。
2. Mac 把 speech chunk 通过 WebSocket `audio` 发到 GPU。
3. GPU 的 FunASR streaming 持续返回 `asr_partial`。
4. 如果当前状态是 `BOT_SPEAKING`，Interrupt-VAD 会用 partial text 判断是否真打断：
   - “嗯/好/对”等短反馈默认视为假打断，不停播。
   - “等等/停/不是/我想...”等明确修正或较长新意图视为真打断。
5. 真打断时状态进入 `USER_INTERRUPTING`，服务调用 vLLM `abort(request_id)`，同时设置 TTS cancel event，Mac 收到 `interrupt` 后清空本地播放队列。
6. Mac 在本地 VAD 判断一句话结束后发送 `end_utterance`。
7. GPU 状态进入 `THINKING`，vLLM 用 Qwen2.5-7B 生成，边生成边返回 `llm_token`。
8. 服务把 token 累积到短句或标点后送入 CosyVoice streaming，返回 `tts_audio` chunk。
9. 第一块 TTS 音频出来后状态保持 `BOT_SPEAKING`，Mac 边收边播。
10. 回复结束后发送 `bot_final`，状态回到 `LISTENING`。

## 关键实现位置

- [gpu_server/app/main.py](/Users/liutao1102c/Downloads/CallRobot/gpu_server/app/main.py): WebSocket 编排、取消路径、ASR/LLM/TTS 串流。
- [gpu_server/app/services/llm_vllm.py](/Users/liutao1102c/Downloads/CallRobot/gpu_server/app/services/llm_vllm.py): vLLM streaming 和 request abort。
- [gpu_server/app/services/interrupt_vad.py](/Users/liutao1102c/Downloads/CallRobot/gpu_server/app/services/interrupt_vad.py): 上下文打断判断。
- [gpu_server/app/services/state_manager.py](/Users/liutao1102c/Downloads/CallRobot/gpu_server/app/services/state_manager.py): 会话状态机。
- [gpu_server/app/services/asr_funasr.py](/Users/liutao1102c/Downloads/CallRobot/gpu_server/app/services/asr_funasr.py): FunASR streaming cache。
- [gpu_server/app/services/tts_cosyvoice.py](/Users/liutao1102c/Downloads/CallRobot/gpu_server/app/services/tts_cosyvoice.py): CosyVoice chunk 输出。
- [mac_client/callrobot_mac/client.py](/Users/liutao1102c/Downloads/CallRobot/mac_client/callrobot_mac/client.py): Mac 测试客户端主入口。
- [mac_client/callrobot_mac/vad.py](/Users/liutao1102c/Downloads/CallRobot/mac_client/callrobot_mac/vad.py): 本地 Fast VAD 和 endpointing。
- [mac_client/callrobot_mac/audio_io.py](/Users/liutao1102c/Downloads/CallRobot/mac_client/callrobot_mac/audio_io.py): 麦克风采集和 TTS 播放队列。

## 延迟建议

- Mac -> GPU 音频 chunk 建议 20-40 ms；FunASR 可按 200-600 ms 聚合送推理，避免过度调用。
- vLLM 建议独占 GPU；如果和 ASR/TTS 共卡，先把 Qwen 的 `QWEN_GPU_MEMORY_UTILIZATION` 调低到 `0.75-0.82`。
- 播放端必须支持队列清空：收到 `interrupt` 或 `cancelled` 立即停止当前音频并丢弃已缓存 chunk。
- 生产环境建议把 ASR、LLM、TTS 拆成独立进程或服务，本工程先给出单服务编排，便于跑通端到端链路。
