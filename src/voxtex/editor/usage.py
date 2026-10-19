"""The usage panel (ctrl+g): small graphs for what voxtex is using right now, in herdr's
sidebar style. Each section appears only when it applies: `ai` once a model is in the
chain, `gpu` when speech or the local model runs on an NVIDIA GPU."""

import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field

from rich.table import Table
from rich.text import Text

from voxtex.editor.palette import VOXTEX, Palette

SPARKS = "▁▂▃▄▅▆▇█"
HISTORY = 30  # samples kept for each graph
GPU_EVERY_S = 1.5


def spark(values: list[float], width: int, top: float | None = None) -> str:
    """The last `width` values as a sparkline, scaled to `top` (default: their maximum)."""
    values = list(values)[-width:]
    if not values:
        return " " * width
    top = top or max(values) or 1.0
    line = "".join(SPARKS[min(7, int(v / top * 7.999))] if v > 0 else " " for v in values)
    return line.rjust(width)


def meter(fraction: float, width: int, pal: Palette = VOXTEX) -> Text:
    """A thin bar, herdr's `━━━───`."""
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return Text.assemble(("━" * filled, pal.accent), ("─" * (width - filled), pal.surface1))


def short(n: float) -> str:
    return f"{n / 1000:.1f}K" if n >= 1000 else f"{n:.0f}"


@dataclass
class GpuSample:
    name: str
    load: float  # 0..100
    used_mb: float
    total_mb: float


@dataclass
class Usage:
    """What the panel shows. The editor feeds it; a thread samples the GPU while shown."""

    tokens_in: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    latency: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    speech: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    gpu_load: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    gpu: GpuSample | None = None
    gpu_used: bool = False  # speech or the AI model runs on the GPU
    _sampler: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def start_gpu(self) -> None:
        if self._sampler is None and shutil.which("nvidia-smi"):
            self._sampler = threading.Thread(target=self._sample, daemon=True)
            self._sampler.start()

    def stop(self) -> None:
        self._stop.set()

    def _sample(self) -> None:
        while not self._stop.is_set():
            sample = read_gpu()
            if sample is None:
                return
            self.gpu = sample
            self.gpu_load.append(sample.load)
            self._stop.wait(GPU_EVERY_S)


def read_gpu() -> GpuSample | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout  # fmt: skip
        name, load, used, total = (x.strip() for x in out.splitlines()[0].split(","))
        return GpuSample(name.removeprefix("NVIDIA ").removeprefix("GeForce "),
                         float(load), float(used), float(total))  # fmt: skip
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def render(usage: Usage, chain, width: int, pal: Palette = VOXTEX) -> Table:
    """The panel's content for a column `width` cells wide."""
    graph = max(8, width - 14)
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(style=pal.overlay0, no_wrap=True, width=8)
    t.add_column(no_wrap=True, justify="right")

    def heading(title: str, detail: str = "") -> None:
        if t.row_count:
            t.add_row("", "")
        t.add_row(Text(title, style=f"bold {pal.text}"), Text(detail, style=pal.overlay0))

    if chain is not None and chain.providers:
        totals = chain.totals
        heading("ai", chain.active or "not used yet")
        t.add_row("calls", Text(str(totals.calls), style=pal.text))
        t.add_row(
            "tokens",
            Text.assemble((short(totals.input_tokens), pal.text), (" in · ", pal.overlay0),
                          (short(totals.output_tokens), pal.text), (" out", pal.overlay0)),
        )  # fmt: skip
        if usage.tokens_in:
            t.add_row("per call", Text(spark(usage.tokens_in, graph), style=pal.mauve))
        if usage.latency:
            last = usage.latency[-1]
            t.add_row("latency", Text.assemble((spark(usage.latency, graph - 7), pal.yellow),
                                               (f" {last:4.1f}s", pal.text)))  # fmt: skip
        left = chain.quota.remaining_tokens(chain.active) if chain.active else None
        if left is not None:
            t.add_row("quota", Text.assemble((f"~{short(left)} tokens left", pal.text)))

    if usage.speech:
        heading("speech")
        last = usage.speech[-1]
        t.add_row("time", Text.assemble((spark(usage.speech, graph - 7), pal.green),
                                        (f" {last:4.1f}s", pal.text)))  # fmt: skip

    if usage.gpu_used and usage.gpu is not None:
        g = usage.gpu
        heading("gpu", g.name)
        t.add_row("load", Text.assemble((spark(usage.gpu_load, graph - 5, top=100), pal.peach),
                                        (f" {g.load:3.0f}%", pal.text)))  # fmt: skip
        frac = g.used_mb / g.total_mb if g.total_mb else 0
        t.add_row("memory", Text.assemble(meter(frac, graph - 9, pal),
                                          (f" {g.used_mb / 1024:.1f}/{g.total_mb / 1024:.0f}G",
                                           pal.text)))  # fmt: skip
    if not t.row_count:
        t.add_row(Text("nothing to show yet", style=pal.overlay0), "")
    return t
