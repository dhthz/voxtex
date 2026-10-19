"""Microphone → utterances, using WebRTC voice activity detection.

An utterance ends after `pause_s` of silence (0.7 s by default) and goes to
speech-to-text right away; whether the *thought* is finished is decided later.
"""

import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
START_FRAMES = 5  # voiced frames in a row (150 ms) that start an utterance
PRE_ROLL_MS = 300  # audio kept from before speech started, so first syllables aren't cut


@dataclass
class Utterance:
    audio: np.ndarray  # float32 mono 16 kHz
    started: float
    ended: float  # time of the last voiced frame (= "end of speech")

    @property
    def seconds(self) -> float:
        return len(self.audio) / SAMPLE_RATE


class Segmenter:
    def __init__(
        self,
        is_speech: Callable[[bytes], bool],
        pause_s: float = 0.7,
        min_speech_s: float = 0.4,
        max_utterance_s: float = 30.0,
    ):
        self.is_speech = is_speech
        self.pause_frames = max(1, round(pause_s * 1000 / FRAME_MS))
        self.min_voiced = max(1, round(min_speech_s * 1000 / FRAME_MS))
        self.max_frames = round(max_utterance_s * 1000 / FRAME_MS)
        self.pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_MS // FRAME_MS)
        self.frames: list[np.ndarray] = []
        self.speaking = False
        self.run = 0  # consecutive voiced frames while idle
        self.silent = 0  # consecutive silent frames while speaking
        self.voiced = 0
        self.started = 0.0
        self.last_voice = 0.0

    def feed(self, frame: np.ndarray, now: float) -> Utterance | None:
        """`frame`: FRAME_SAMPLES int16 samples."""
        voiced = self.is_speech(frame.tobytes())
        if voiced:
            self.last_voice = now
        if not self.speaking:
            self.pre_roll.append(frame)
            self.run = self.run + 1 if voiced else 0
            if self.run >= START_FRAMES:
                self.speaking = True
                self.frames = list(self.pre_roll)
                self.voiced = self.run
                self.silent = 0
                self.started = now
            return None
        self.frames.append(frame)
        if voiced:
            self.voiced += 1
            self.silent = 0
        else:
            self.silent += 1
        if self.silent >= self.pause_frames or len(self.frames) >= self.max_frames:
            return self._finish()
        return None

    def _finish(self) -> Utterance | None:
        frames, voiced = self.frames, self.voiced
        self.speaking, self.frames, self.run, self.voiced, self.silent = False, [], 0, 0, 0
        self.pre_roll.clear()
        if voiced < self.min_voiced:
            return None  # a click or a cough
        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        return Utterance(audio, self.started, self.last_voice)


class NoiseGate:
    """Tracks the room's background level and only lets through frames clearly louder
    than it. WebRTC's VAD on its own takes fans, typing and music for speech.

    The background is a low percentile of the last few seconds: steady noise sits there,
    while speech, which keeps stopping between words, never does."""

    def __init__(self, ratio: float = 3.0, window_s: float = 4.0, min_floor: float = 0.0005):
        self.ratio = ratio
        self.min_floor = min_floor
        self.recent: deque[float] = deque(maxlen=round(window_s * 1000 / FRAME_MS))

    @property
    def floor(self) -> float:
        if len(self.recent) < 10:
            return self.min_floor
        return max(self.min_floor, float(np.percentile(self.recent, 10)))

    def loud_enough(self, frame: bytes) -> bool:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        self.recent.append(rms)
        return rms > self.floor * self.ratio


def webrtc_vad(aggressiveness: int = 3) -> Callable[[bytes], bool]:
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    gate = NoiseGate()
    return lambda frame: gate.loud_enough(frame) and vad.is_speech(frame, SAMPLE_RATE)


# --- the microphone ------------------------------------------------------------------------------


class MicError(Exception):
    pass


def prefer_parec() -> bool:
    """WSLg: PortAudio's ALSA→PulseAudio route often times out; parec talks to Pulse directly."""
    return Path("/mnt/wslg").exists() and shutil.which("parec") is not None


def list_inputs() -> list[tuple[int, str]]:
    import sounddevice as sd

    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def pick_device(wanted: str = "") -> int | None:
    """Device index for `wanted` (index or name part); None = system default.

    On WSL the default ALSA device is often silent while `pulse` reaches the Windows mic.
    """
    try:
        inputs = list_inputs()
    except OSError as e:
        raise MicError(f"audio system unavailable: {e}") from e
    if not inputs:
        raise MicError("no microphone found")
    if wanted:
        if wanted.isdigit() and int(wanted) in {i for i, _ in inputs}:
            return int(wanted)
        for i, name in inputs:
            if wanted.lower() in name.lower():
                return i
        raise MicError(f"no microphone matching {wanted!r}")
    for i, name in inputs:
        if name.lower() == "pulse":
            return i
    return None


class Microphone:
    """Captures 30 ms frames on a background thread and turns them into utterances."""

    def __init__(
        self,
        segmenter: Segmenter,
        device: int | None = None,
        backend: str = "portaudio",
        source: str = "",
    ):
        """backend: portaudio (sounddevice) or parec (PulseAudio's recorder; used on WSL,
        where the Windows mic arrives through WSLg's PulseAudio server)."""
        self.segmenter = segmenter
        self.device = device
        self.backend = backend
        self.source = source  # PulseAudio source name for parec ("" = default)
        self._proc: subprocess.Popen | None = None
        self.utterances: queue.Queue[Utterance] = queue.Queue()
        self.level = 0.0  # recent loudness 0..1, for the status line
        self.peak = 0.0  # loudest frame so far (RMS), for `doctor`
        self.recording: list[np.ndarray] | None = None  # set to [] to also keep raw frames
        self._frames: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._rate = SAMPLE_RATE
        self._carry = np.zeros(0, dtype=np.int16)

    @property
    def speaking(self) -> bool:
        return self.segmenter.speaking

    def start(self) -> None:
        if self.backend == "parec":
            self._start_parec()
            return
        try:
            self._start_portaudio()
        except MicError:
            if not shutil.which("parec"):
                raise
            self.backend = "parec"  # PulseAudio directly (works where PortAudio's route doesn't)
            self._start_parec()

    def _start_portaudio(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, t, status):
            self._frames.put(np.frombuffer(indata, dtype=np.int16).copy())

        try:
            self._stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES,
                device=self.device, callback=callback,
            )  # fmt: skip
            self._stream.start()
        except Exception:
            # Some devices only do their native rate: capture at that rate and resample.
            try:
                info = sd.query_devices(self.device, "input")
                self._rate = int(info["default_samplerate"])
                self._stream = sd.RawInputStream(
                    samplerate=self._rate, channels=1, dtype="int16",
                    blocksize=self._rate * FRAME_MS // 1000, device=self.device, callback=callback,
                )  # fmt: skip
                self._stream.start()
            except Exception as e:
                raise MicError(f"cannot open the microphone: {e}") from e
        self._thread.start()

    def _start_parec(self) -> None:
        cmd = ["parec", "--format=s16le", f"--rate={SAMPLE_RATE}", "--channels=1", "--raw",
               f"--latency-msec={FRAME_MS}"]  # fmt: skip
        if self.source:
            cmd.append(f"--device={self.source}")
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError as e:
            raise MicError(f"cannot start parec: {e}") from e

        def reader() -> None:
            chunk = FRAME_SAMPLES * 2
            while not self._stop.is_set():
                data = self._proc.stdout.read(chunk)
                if not data:
                    break
                self._frames.put(np.frombuffer(data, dtype=np.int16).copy())

        threading.Thread(target=reader, daemon=True).start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self._proc is not None:
            self._proc.terminate()

    def _resample(self, chunk: np.ndarray) -> np.ndarray:
        if self._rate == SAMPLE_RATE:
            return chunk
        n = round(len(chunk) * SAMPLE_RATE / self._rate)
        x = np.linspace(0, len(chunk) - 1, n)
        return np.interp(x, np.arange(len(chunk)), chunk).astype(np.int16)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._frames.get(timeout=0.2)
            except queue.Empty:
                continue
            data = np.concatenate([self._carry, self._resample(chunk)])
            while len(data) >= FRAME_SAMPLES:
                frame, data = data[:FRAME_SAMPLES], data[FRAME_SAMPLES:]
                rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768) ** 2)))
                self.level = 0.8 * self.level + 0.2 * min(1.0, rms * 8)
                self.peak = max(self.peak, rms)
                if self.recording is not None:
                    self.recording.append(frame)
                u = self.segmenter.feed(frame, time.monotonic())
                if u is not None:
                    self.utterances.put(u)
            self._carry = data


def open_microphone(segmenter: Segmenter, wanted: str = "") -> Microphone:
    """The microphone for `wanted` (index, name part or PulseAudio source; "" = default)."""
    if prefer_parec() and not wanted.isdigit():
        return Microphone(segmenter, backend="parec", source=wanted)
    return Microphone(segmenter, pick_device(wanted))
