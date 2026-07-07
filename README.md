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
%env FUNASR_VAD_MODEL=/kaggle/input/callrobot-models/speech_fsmn_vad_zh-cn-16k-common-pytorch
%env FUNASR_PUNC_MODEL=/kaggle/input/callrobot-models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
%env COSYVOICE_MODEL=/kaggle/input/callrobot-models/CosyVoice-300M-SFT
%env COSYVOICE_REPO_DIR=/kaggle/working/CosyVoice
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
