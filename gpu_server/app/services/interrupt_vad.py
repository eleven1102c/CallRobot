from dataclasses import dataclass

from gpu_server.app.config import Settings
from gpu_server.app.models.protocol import DialogueState
from gpu_server.app.services.state_manager import DialogueSession


@dataclass(frozen=True)
class InterruptDecision:
    is_interrupt: bool
    confidence: float
    reason: str


class InterruptVAD:
    """Context-aware interrupt classifier.

    Fast VAD runs on the Mac and only tells the GPU side that user speech exists.
    This layer decides whether that speech should stop the bot, using dialogue
    state plus partial ASR text. It is intentionally conservative for short
    backchannels such as "嗯", "好", "对".
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backchannels = {"嗯", "啊", "哦", "好", "对", "是", "行", "可以", "好的"}
        self.command_words = {"停", "停止", "等等", "等一下", "打断", "别说了", "换一个", "不是"}

    def decide(self, session: DialogueSession, partial_text: str) -> InterruptDecision:
        text = partial_text.strip()
        if session.state != DialogueState.BOT_SPEAKING:
            return InterruptDecision(False, 0.0, "bot_not_speaking")

        if not text:
            return InterruptDecision(False, 0.15, "speech_without_asr_text")

        if any(word in text for word in self.command_words):
            return InterruptDecision(True, 0.95, "explicit_interrupt_command")

        normalized = text.replace("，", "").replace(",", "").replace("。", "")
        if normalized in self.backchannels:
            return InterruptDecision(False, 0.25, "short_backchannel")

        score = 0.35
        if len(text) >= self.settings.false_interrupt_min_chars:
            score += 0.18
        if len(text) >= 5:
            score += 0.18
        if any(mark in text for mark in ("?", "？", "不对", "不是", "我想", "你先")):
            score += 0.22
        if session.bot_text_buffer[-self.settings.interrupt_recent_bot_chars:] and text not in session.bot_text_buffer:
            score += 0.08

        is_interrupt = score >= self.settings.interrupt_confidence_threshold
        return InterruptDecision(is_interrupt, min(score, 0.99), "context_score")
