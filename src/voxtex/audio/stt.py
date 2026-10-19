"""Speech-to-text: local faster-whisper or Groq/OpenAI Whisper, chosen to suit the machine.

`auto` (the default):
  NVIDIA GPU      → local large-v3-turbo (fast, accurate, private)
  no GPU + key    → Groq whisper-large-v3-turbo (fast, accurate; audio goes to the cloud)
  no GPU, no key  → local `small` on the CPU (slower, weaker in Greek)
`--private` always means local. Models are always multilingual (never `.en`).
"""

import contextlib
import ctypes
import gc
import glob
import io
import os
import site
import time
import wave
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from voxtex.config import Config

SAMPLE_RATE = 16000
PROMPT_MAX_CHARS = 400
# Whisper's usual inventions on near-silence.
_HALLUCINATION_NO_SPEECH = 0.6
_HALLUCINATION_LOGPROB = -1.0
# Switching language between utterances needs a clear signal: short chunks are often
# misjudged, and a wrong language garbles the whole sentence.
SWITCH_MIN_SECONDS = 1.5
SWITCH_MIN_SHARE = 0.75  # of the probability among the allowed languages
# Below this average log-probability Whisper was guessing (on a short clip, often because it
# picked the wrong language): the other allowed languages get a try and the surest one wins.
RECHECK_LOGPROB = -0.7
NOTHING_HEARD = -10.0


@dataclass
class Transcript:
    text: str
    seconds: float  # time spent transcribing
    audio_seconds: float
    backend: str
    language: str = ""
    confidence: float = 0.0  # Whisper's average log-probability per token (0 = not known)


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, prompt: str = "") -> Transcript: ...


@dataclass
class SttChoice:
    kind: str  # local | groq | openai
    model: str
    device: str = "cpu"  # cuda | cpu (local only)
    reason: str = ""

    @property
    def label(self) -> str:
        where = f" on {self.device.upper()}" if self.kind == "local" else " (cloud)"
        return f"{self.kind} {self.model}{where}"


# --- CUDA ----------------------------------------------------------------------------------------


def _preload_cuda_libs() -> None:
    """pip's nvidia-* wheels aren't on the loader path; load them so CTranslate2 finds them."""
    dirs = []
    for sp in [*site.getsitepackages(), site.getusersitepackages()]:
        dirs += glob.glob(os.path.join(sp, "nvidia", "cublas", "lib"))
        dirs += glob.glob(os.path.join(sp, "nvidia", "cudnn", "lib"))
    for d in dirs:
        for lib in sorted(glob.glob(os.path.join(d, "lib*.so*")), key=lambda p: "Lt" not in p):
            with contextlib.suppress(OSError):
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)


_cuda: bool | None = None


def cuda_available() -> bool:
    global _cuda
    if _cuda is None:
        try:
            _preload_cuda_libs()
            import ctranslate2

            _cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            _cuda = False
    return _cuda


# --- choosing ------------------------------------------------------------------------------------


def choose_stt(cfg: Config, private: bool = False, mode: str | None = None) -> SttChoice:
    mode = mode or cfg.stt_mode
    gpu = cuda_available()
    local_model = cfg.stt_local_model
    if local_model == "auto":
        local_model = "large-v3-turbo" if gpu else "small"
    local_model = local_model.removesuffix(".en")  # Greek needs a multilingual model
    local = SttChoice("local", local_model, "cuda" if gpu else "cpu")
    groq = cfg.providers.get("groq-gptoss")
    groq_key = groq.resolve_key() if groq else None

    if private or mode == "local":
        local.reason = "private mode" if private else "configured"
        return local
    if mode in ("groq", "openai"):
        return SttChoice(
            mode, cfg.stt_cloud_model if mode == "groq" else "whisper-1", reason="configured"
        )
    if gpu:
        local.reason = "NVIDIA GPU found"
        return local
    if groq_key:
        return SttChoice("groq", cfg.stt_cloud_model, reason="no GPU; Groq key found")
    local.reason = "no GPU and no Groq key (slower; weaker in Greek)"
    return local


def build_transcriber(
    choice: SttChoice, cfg: Config, languages: tuple[str, ...] = ()
) -> Transcriber:
    if choice.kind == "local":
        return LocalWhisper(choice.model, choice.device, cfg.stt_language, languages)
    p = cfg.providers["groq-gptoss"] if choice.kind == "groq" else None
    base_url = p.base_url if p else "https://api.openai.com/v1"
    key = (p.resolve_key() if p else os.environ.get("OPENAI_API_KEY")) or ""
    return CloudWhisper(choice.model, base_url, key, cfg.stt_language, name=choice.kind)


# --- backends ------------------------------------------------------------------------------------


class LocalWhisper:
    def __init__(
        self,
        model: str,
        device: str = "cpu",
        language: str = "auto",
        languages: tuple[str, ...] = (),
    ):
        """`languages`: with language "auto", detection only chooses among these (e.g. el, en),
        which stops Whisper inventing Chinese or Russian text from background noise."""
        self.name = f"local {model}"
        self.model_name = model
        self.device = device
        self.language = None if language == "auto" else language
        self.languages = languages
        self.last_language = languages[0] if languages else None
        self._model = None

    def load(self) -> None:
        if self._model is None:
            if self.device == "cuda":
                _preload_cuda_libs()
            from faster_whisper import WhisperModel

            compute = "float16" if self.device == "cuda" else "int8"
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=compute)
            # The first GPU call is slow (kernel setup); pay it now, not on the first sentence.
            self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def unload(self) -> None:
        """Release the model from the main thread (avoids a CUDA abort at interpreter exit)."""
        self._model = None
        gc.collect()

    def _pick_language(self, encoder_output, seconds: float) -> str | None:
        probs = self._model.model.detect_language(encoder_output)[0]
        ranked = [(tok[2:-2], p) for tok, p in probs]
        allowed = [(lang, p) for lang, p in ranked if lang in self.languages]
        if not allowed:
            return ranked[0][0] if ranked else None
        return self._sticky(allowed, seconds)

    def _sticky(self, allowed: list[tuple[str, float]], seconds: float) -> str:
        """The likeliest allowed language, but stay with the last one unless the evidence
        for a switch is clear (a long enough chunk, most of the probability)."""
        lang, p = max(allowed, key=lambda lp: lp[1])
        share = p / (sum(q for _, q in allowed) or 1)
        last = self.last_language
        if last and lang != last and (seconds < SWITCH_MIN_SECONDS or share < SWITCH_MIN_SHARE):
            lang = last
        self.last_language = lang
        return lang

    def transcribe(self, audio: np.ndarray, prompt: str = "") -> Transcript:
        self.load()
        m = self._model
        start = time.monotonic()
        language = self.language
        restore = None
        if language is None and self.languages:
            # One encoder pass for both language detection and transcription (auto-detect
            # would run the encoder twice, ~0.7 s extra per utterance on a mid-range GPU).
            from faster_whisper.audio import pad_or_trim

            n = m.feature_extractor.nb_max_frames
            features = m.feature_extractor(audio)
            encoded = m.encode(pad_or_trim(features[..., :n]))
            language = self._pick_language(encoded, len(audio) / SAMPLE_RATE)
            if features.shape[-1] < n:  # a single 30 s window: reuse it
                calls = {"n": 0}
                original = m.encode

                def encode_once(segment):
                    calls["n"] += 1
                    return encoded if calls["n"] == 1 else original(segment)

                m.encode, restore = encode_once, original
        try:
            text, confidence, detected = self._decode(audio, language, prompt)
        finally:
            if restore is not None:
                m.encode = restore
        if self.language is None and confidence < RECHECK_LOGPROB:
            for other in self.languages:
                if other == language:
                    continue
                t, c, _ = self._decode(audio, other, prompt)
                if c > confidence:
                    text, confidence, language = t, c, other
                    self.last_language = other
        return Transcript(
            text,
            time.monotonic() - start,
            len(audio) / SAMPLE_RATE,
            self.name,
            language or detected,
            confidence,
        )

    def _decode(self, audio: np.ndarray, language: str | None, prompt: str):
        """(text, average log-probability, detected language) for one pass of Whisper."""
        segments, info = self._model.transcribe(
            audio,
            language=language,
            initial_prompt=prompt[:PROMPT_MAX_CHARS] or None,
            beam_size=5 if self.device == "cuda" else 1,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,  # we already cut utterances with our own VAD
        )
        kept = [s for s in segments if not _invented(s)]
        if not kept:
            return "", NOTHING_HEARD, info.language
        tokens = sum(max(1, len(s.tokens)) for s in kept)
        confidence = sum(s.avg_logprob * max(1, len(s.tokens)) for s in kept) / tokens
        return "".join(s.text for s in kept).strip(), confidence, info.language


def _invented(segment) -> bool:
    """Whisper's output on noise: unsure there was speech and unsure of the words, or the
    same phrase looping."""
    unsure = (
        segment.no_speech_prob > _HALLUCINATION_NO_SPEECH
        and segment.avg_logprob < _HALLUCINATION_LOGPROB
    )
    return unsure or segment.no_speech_prob > 0.9 or segment.compression_ratio > 2.4


def to_wav(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


class CloudWhisper:
    def __init__(
        self, model: str, base_url: str, key: str, language: str = "auto", name: str = "groq"
    ):
        import openai

        self.name = f"{name} {model}"
        self.model = model
        self.language = None if language == "auto" else language
        self.client = openai.OpenAI(base_url=base_url, api_key=key, timeout=15, max_retries=1)

    def transcribe(self, audio: np.ndarray, prompt: str = "") -> Transcript:
        start = time.monotonic()
        kwargs: dict = {
            "model": self.model,
            "file": ("utterance.wav", to_wav(audio)),
            "temperature": 0,
        }
        if prompt:
            kwargs["prompt"] = prompt[:PROMPT_MAX_CHARS]
        if self.language:
            kwargs["language"] = self.language
        result = self.client.audio.transcriptions.create(**kwargs)
        return Transcript(
            (result.text or "").strip(),
            time.monotonic() - start,
            len(audio) / SAMPLE_RATE,
            self.name,
        )
