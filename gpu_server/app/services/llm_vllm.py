from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from gpu_server.app.config import Settings


class StreamingLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: Any | None = None
        self.tokenizer: Any | None = None

    async def load(self) -> None:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.qwen_model, trust_remote_code=True)
        engine_kwargs = {
            "model": self.settings.qwen_model,
            "tensor_parallel_size": self.settings.qwen_tensor_parallel_size,
            "gpu_memory_utilization": self.settings.qwen_gpu_memory_utilization,
            "max_model_len": self.settings.qwen_max_model_len,
            "dtype": self.settings.qwen_dtype,
            "max_num_seqs": self.settings.qwen_max_num_seqs,
            "max_num_batched_tokens": self.settings.qwen_max_num_batched_tokens,
            "cpu_offload_gb": self.settings.qwen_cpu_offload_gb,
            "swap_space": self.settings.qwen_swap_space,
            "trust_remote_code": True,
            "enforce_eager": self.settings.qwen_enforce_eager,
        }
        if self.settings.qwen_quantization:
            engine_kwargs["quantization"] = self.settings.qwen_quantization
        args = AsyncEngineArgs(**engine_kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)

    def _prompt(self, history: list[dict[str, str]], user_text: str) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded")
        messages = [
            {"role": "system", "content": "你是一个低延迟全双工语音助手。回答要自然、简洁，可被直接朗读。"}
        ]
        messages.extend(history[-12:])
        messages.append({"role": "user", "content": user_text})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    async def stream(self, request_id: str, history: list[dict[str, str]], user_text: str) -> AsyncIterator[str]:
        if self.engine is None:
            raise RuntimeError("StreamingLLM is not loaded")

        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=self.settings.qwen_temperature,
            top_p=self.settings.qwen_top_p,
            max_tokens=self.settings.qwen_max_tokens,
        )
        last_text = ""
        async for output in self.engine.generate(self._prompt(history, user_text), sampling, request_id):
            text = output.outputs[0].text
            delta = text[len(last_text):]
            last_text = text
            if delta:
                yield delta

    async def cancel(self, request_id: str | None) -> None:
        if request_id and self.engine is not None:
            await self.engine.abort(request_id)
