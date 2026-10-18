"""OpenAI-compatible providers chained with failover (addendum §4).

The chain tries providers in order. Rate limits mark a provider exhausted until
its reset time (persisted in .voxtex/quota.json); timeouts/5xx get one retry;
unparseable output gets one retry on the same provider. All of this happens
inside a single `complete()` call, so the caller never loses a request.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

import openai

from voxtex.config import ProviderConfig

DEFAULT_EXHAUSTED_S = 3600.0
MAX_OUTPUT_TOKENS = 1500

T = TypeVar("T")


# --- errors raised by a `send` function --------------------------------------------------------


class ProviderError(Exception):
    reason = "error"


class RateLimited(ProviderError):
    reason = "quota reached"

    def __init__(self, reset_s: float | None = None):
        super().__init__(f"rate limited (reset in {reset_s}s)")
        self.reset_s = reset_s


class Transient(ProviderError):
    """Timeout or 5xx: worth one retry."""

    reason = "not responding"


class JsonModeUnsupported(ProviderError):
    reason = "no JSON mode"


class Unavailable(ProviderError):
    """Bad key, unknown model, server not running: skip for the rest of the session."""

    reason = "unavailable"


class ParseError(Exception):
    """Raised by a chain `parse` callback when the model's output is unusable."""


class ChainExhausted(Exception):
    pass


@dataclass
class Reply:
    text: str
    headers: dict[str, str] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


Send = Callable[[ProviderConfig, str, list[dict], bool, float], Reply]


# --- the real transport --------------------------------------------------------------------------


def parse_duration(s: str | None) -> float | None:
    """Parse '7.66s', '2m59.56s', '1h2m', '500ms' or a bare number of seconds."""
    if not s:
        return None
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    total, found = 0.0, False
    for num, unit in re.findall(r"([\d.]+)\s*(ms|h|m|s)", s):
        total += float(num) * {"ms": 0.001, "h": 3600, "m": 60, "s": 1}[unit]
        found = True
    return total if found else None


def reset_seconds(headers: dict[str, str]) -> float | None:
    h = {k.lower(): v for k, v in headers.items()}
    if (ra := parse_duration(h.get("retry-after"))) is not None:
        return ra
    resets = [
        parse_duration(h.get(k)) for k in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")
    ]
    resets = [r for r in resets if r is not None]
    return max(resets) if resets else None


def send_openai(
    p: ProviderConfig, key: str, messages: list[dict], json_mode: bool, timeout: float
) -> Reply:
    client = openai.OpenAI(
        base_url=p.base_url, api_key=key or "not-needed", timeout=timeout, max_retries=0
    )
    kwargs: dict = {"model": p.model, "messages": messages, "temperature": 0.2}
    kwargs["max_tokens"] = MAX_OUTPUT_TOKENS
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if p.reasoning_effort:
        kwargs["reasoning_effort"] = p.reasoning_effort
    try:
        try:
            raw = client.chat.completions.with_raw_response.create(**kwargs)
        except openai.BadRequestError as e:
            if "reasoning_effort" not in kwargs or "reasoning" not in str(e).lower():
                raise
            kwargs.pop("reasoning_effort")  # the model doesn't support it: go without
            raw = client.chat.completions.with_raw_response.create(**kwargs)
    except openai.RateLimitError as e:
        raise RateLimited(reset_seconds(dict(e.response.headers))) from e
    except openai.APITimeoutError as e:
        raise Transient("timeout") from e
    except openai.APIConnectionError as e:
        raise Unavailable(f"cannot connect to {p.base_url}") from e
    except openai.InternalServerError as e:
        raise Transient(str(e)) from e
    except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as e:
        raise Unavailable(str(e)) from e
    except openai.BadRequestError as e:
        if json_mode and re.search(r"response_format|json", str(e), re.I):
            raise JsonModeUnsupported(str(e)) from e
        raise Unavailable(str(e)) from e
    except openai.APIStatusError as e:
        if e.status_code >= 500:
            raise Transient(str(e)) from e
        raise Unavailable(str(e)) from e
    resp = raw.parse()
    usage = resp.usage
    return Reply(
        text=resp.choices[0].message.content or "",
        headers=dict(raw.headers),
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


# --- quota state ---------------------------------------------------------------------------------


class QuotaStore:
    """Per-provider exhaustion and remaining-quota info, persisted as JSON."""

    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict[str, dict] = {}
        if path and path.is_file():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.data = {}

    def exhausted(self, name: str, now: float) -> bool:
        return self.data.get(name, {}).get("exhausted_until", 0) > now

    def mark_exhausted(self, name: str, until: float) -> None:
        self.data.setdefault(name, {})["exhausted_until"] = until
        self._save()

    def update(self, name: str, headers: dict[str, str]) -> None:
        h = {k.lower(): v for k, v in headers.items()}
        entry = self.data.setdefault(name, {})
        for key in ("remaining-tokens", "remaining-requests"):
            v = h.get(f"x-ratelimit-{key}")
            if v and v.isdigit():
                entry[key.replace("-", "_")] = int(v)
        if entry:
            self._save()

    def remaining_tokens(self, name: str) -> int | None:
        return self.data.get(name, {}).get("remaining_tokens")

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        except OSError:
            pass


# --- the chain -----------------------------------------------------------------------------------


@dataclass
class ChainResult(Generic[T]):
    value: T
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    notices: list[str] = field(default_factory=list)


@dataclass
class Totals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class Chain:
    def __init__(
        self,
        providers: list[ProviderConfig],
        default_timeout: float = 8.0,
        quota: QuotaStore | None = None,
        log_path: Path | None = None,
        send: Send | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.providers = providers
        self.default_timeout = default_timeout
        self.quota = quota or QuotaStore(None)
        self.log_path = log_path
        self.send = send or send_openai
        self.clock = clock
        self.disabled: dict[str, str] = {}  # name -> reason, for this session
        self.json_off: set[str] = set()
        self.active: str | None = None
        self.totals = Totals()

    def usable(self) -> list[ProviderConfig]:
        now = self.clock()
        return [
            p
            for p in self.providers
            if p.resolve_key() is not None
            and p.name not in self.disabled
            and not self.quota.exhausted(p.name, now)
        ]

    def complete(self, messages: list[dict], parse: Callable[[str], T]) -> ChainResult[T]:
        failures: list[tuple[str, str]] = []
        for p in self.usable():
            key = p.resolve_key() or ""
            timeout = p.timeout or self.default_timeout
            retried_transient = retried_parse = False
            while True:
                json_mode = p.json_mode and p.name not in self.json_off
                start = self.clock()
                try:
                    reply = self.send(p, key, messages, json_mode, timeout)
                except RateLimited as e:
                    reset = e.reset_s if e.reset_s is not None else DEFAULT_EXHAUSTED_S
                    self.quota.mark_exhausted(p.name, self.clock() + reset)
                    failures.append((p.name, e.reason))
                    break
                except Transient as e:
                    if not retried_transient:
                        retried_transient = True
                        continue
                    failures.append((p.name, e.reason))
                    break
                except JsonModeUnsupported as e:
                    if not json_mode:  # already off: don't loop
                        self.disabled[p.name] = str(e)
                        failures.append((p.name, e.reason))
                        break
                    self.json_off.add(p.name)
                    continue
                except Unavailable as e:
                    self.disabled[p.name] = str(e)
                    failures.append((p.name, e.reason))
                    break
                latency = self.clock() - start
                self.quota.update(p.name, reply.headers)
                self._record(p, reply, latency)
                try:
                    value = parse(reply.text)
                except ParseError:
                    if not retried_parse:
                        retried_parse = True
                        continue
                    failures.append((p.name, "unusable output"))
                    break
                notices = []
                if self.active is not None and self.active != p.name:
                    why = f" ({failures[0][0]} {failures[0][1]})" if failures else ""
                    notices.append(f"↪ switched to {p.name}{why}")
                self.active = p.name
                return ChainResult(
                    value=value,
                    provider=p.name,
                    model=p.model,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                    latency_s=latency,
                    notices=notices,
                )
        if failures:
            detail = "; ".join(f"{n}: {r}" for n, r in failures)
            raise ChainExhausted(f"every provider failed ({detail})")
        raise ChainExhausted(
            "no usable provider: set an API key (e.g. GROQ_API_KEY) or start Ollama"
        )

    def _record(self, p: ProviderConfig, reply: Reply, latency: float) -> None:
        self.totals.calls += 1
        self.totals.input_tokens += reply.input_tokens
        self.totals.output_tokens += reply.output_tokens
        if not self.log_path:
            return
        entry = {
            "ts": round(self.clock(), 3),
            "kind": "llm",
            "provider": p.name,
            "model": p.model,
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
            "latency_ms": round(latency * 1000),
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass
