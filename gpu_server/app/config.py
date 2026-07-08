from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 9000
    sample_rate: int = 16000

    qwen_model: str = "/models/Qwen2.5-7B-Instruct"
    qwen_tensor_parallel_size: int = 1
    qwen_gpu_memory_utilization: float = 0.72
    qwen_max_model_len: int = 4096
    qwen_max_tokens: int = 256
    qwen_dtype: str = "float16"
    qwen_quantization: str | None = None
    qwen_max_num_seqs: int = 1
    qwen_max_num_batched_tokens: int = 2048
    qwen_cpu_offload_gb: float = 0.0
    qwen_swap_space: int = 2
    qwen_enforce_eager: bool = True
    qwen_temperature: float = 0.7
    qwen_top_p: float = 0.9

    funasr_model: str = "/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
    funasr_use_vad: bool = False
    funasr_vad_model: str | None = "/models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    funasr_punc_model: str | None = None
    funasr_chunk_size: str = "5,10,5"
    funasr_encoder_chunk_look_back: int = 4
    funasr_decoder_chunk_look_back: int = 1

    cosyvoice_model: str = "/models/CosyVoice-300M-SFT"
    cosyvoice_repo_dir: str | None = None
    cosyvoice_mode: str = "sft"
    cosyvoice_spk: str = "中文女"
    cosyvoice_prompt_wav: str | None = None
    cosyvoice_prompt_text: str | None = None
    cosyvoice_instruct_text: str | None = None
    tts_sample_rate: int = 22050
    tts_flush_chars: int = 8

    false_interrupt_min_chars: int = 2
    interrupt_confidence_threshold: float = 0.62
    interrupt_recent_bot_chars: int = 120

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
