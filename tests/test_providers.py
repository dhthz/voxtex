import json

import pytest

from voxtex.config import ProviderConfig
from voxtex.providers import (
    Chain,
    ChainExhausted,
    JsonModeUnsupported,
    ParseError,
    QuotaStore,
    RateLimited,
    Reply,
    Transient,
    Unavailable,
    parse_duration,
    reset_seconds,
)

CLOUD = "https://api.groq.com/openai/v1"
LOCAL = "http://localhost:11434/v1"


def prov(name, private=False, key="k"):
    return ProviderConfig(
        name=name,
        base_url=LOCAL if private else CLOUD,
        model=f"{name}-model",
        api_key=key,
        private=private,
    )


class FakeSend:
    """Plays back a script of results per provider; records every call."""

    def __init__(self, script: dict[str, list]):
        self.script = script
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, p, key, messages, json_mode, timeout):
        self.calls.append((p.name, json_mode))
        item = self.script[p.name].pop(0)
        if isinstance(item, Exception):
            raise item
        return Reply(text=item, input_tokens=100, output_tokens=10)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def ident(text):
    if text == "bad":
        raise ParseError("bad")
    return text


@pytest.mark.parametrize(
    "s, expected",
    [("7.66s", 7.66), ("2m59.5s", 179.5), ("1h2m", 3720), ("500ms", 0.5), ("30", 30), ("", None)],
)
def test_parse_duration(s, expected):
    assert parse_duration(s) == expected


def test_reset_seconds():
    assert reset_seconds({"Retry-After": "12"}) == 12
    assert (
        reset_seconds({"x-ratelimit-reset-tokens": "7s", "x-ratelimit-reset-requests": "2m"}) == 120
    )
    assert reset_seconds({}) is None


def test_first_provider_success():
    send = FakeSend({"a": ["ok"], "b": []})
    r = Chain([prov("a"), prov("b")], send=send).complete([], ident)
    assert (r.value, r.provider, r.notices) == ("ok", "a", [])


def test_rate_limit_moves_on_and_persists(tmp_path):
    qpath = tmp_path / "quota.json"
    clock = Clock()
    send = FakeSend({"a": [RateLimited(reset_s=60)], "b": ["first", "second"]})
    chain = Chain([prov("a"), prov("b")], quota=QuotaStore(qpath), send=send, clock=clock)
    chain.active = "a"  # pretend 'a' served the previous request
    r = chain.complete([], ident)
    assert (r.value, r.provider) == ("first", "b")
    assert r.notices == ["↪ switched to b (a quota reached)"]
    assert json.loads(qpath.read_text())["a"]["exhausted_until"] == 1060

    # A restart reads the persisted state and skips 'a' without calling it.
    chain2 = Chain([prov("a"), prov("b")], quota=QuotaStore(qpath), send=send, clock=clock)
    assert chain2.complete([], ident).provider == "b"
    assert [c[0] for c in send.calls] == ["a", "b", "b"]

    clock.t = 1061  # after the reset, 'a' is tried again
    assert [p.name for p in chain2.usable()] == ["a", "b"]


def test_rate_limit_without_reset_header_defaults_to_an_hour():
    clock = Clock()
    chain = Chain(
        [prov("a"), prov("b")], send=FakeSend({"a": [RateLimited()], "b": ["ok"]}), clock=clock
    )
    chain.complete([], ident)
    assert chain.quota.data["a"]["exhausted_until"] == 1000 + 3600


def test_transient_retried_once_then_next():
    send = FakeSend({"a": [Transient(), Transient()], "b": ["ok"]})
    assert Chain([prov("a"), prov("b")], send=send).complete([], ident).provider == "b"
    assert [c[0] for c in send.calls] == ["a", "a", "b"]


def test_transient_recovers_on_retry():
    send = FakeSend({"a": [Transient(), "ok"]})
    assert Chain([prov("a")], send=send).complete([], ident).provider == "a"


def test_bad_output_retried_once_on_same_provider_then_next():
    send = FakeSend({"a": ["bad", "bad"], "b": ["ok"]})
    r = Chain([prov("a"), prov("b")], send=send).complete([], ident)
    assert r.provider == "b"
    assert [c[0] for c in send.calls] == ["a", "a", "b"]


def test_json_mode_fallback():
    send = FakeSend({"a": [JsonModeUnsupported(), "ok", "ok2"]})
    chain = Chain([prov("a")], send=send)
    chain.complete([], ident)
    chain.complete([], ident)
    assert send.calls == [("a", True), ("a", False), ("a", False)]


def test_unavailable_disabled_for_session():
    send = FakeSend({"a": [Unavailable("no server")], "b": ["ok", "ok"]})
    chain = Chain([prov("a"), prov("b")], send=send)
    chain.complete([], ident)
    chain.complete([], ident)
    assert [c[0] for c in send.calls] == ["a", "b", "b"]


def test_providers_without_key_are_skipped_silently():
    send = FakeSend({"b": ["ok"]})
    keyless = prov("a").model_copy(update={"api_key": "", "api_key_env": "VOXTEX_TEST_UNSET"})
    assert Chain([keyless, prov("b")], send=send).complete([], ident).provider == "b"
    assert [c[0] for c in send.calls] == ["b"]


def test_everything_fails():
    send = FakeSend({"a": [Unavailable("x")]})
    with pytest.raises(ChainExhausted, match="a: unavailable"):
        Chain([prov("a")], send=send).complete([], ident)
    with pytest.raises(ChainExhausted, match="no usable provider"):
        Chain([], send=send).complete([], ident)


def test_quota_headers_and_totals_and_log(tmp_path):
    log = tmp_path / "timings.log"

    def send(p, key, messages, json_mode, timeout):
        return Reply("ok", {"x-ratelimit-remaining-tokens": "142000"}, 120, 30)

    chain = Chain([prov("a")], send=send, log_path=log)
    chain.complete([], ident)
    chain.complete([], ident)
    assert chain.quota.remaining_tokens("a") == 142000
    assert (chain.totals.calls, chain.totals.input_tokens, chain.totals.output_tokens) == (
        2,
        240,
        60,
    )
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert entries[0]["provider"] == "a" and entries[0]["input_tokens"] == 120


def test_private_chain_never_calls_cloud(tmp_path):
    """Addendum acceptance: --private sends nothing to a non-private provider."""
    from voxtex.config import load_config

    cfg = load_config(None, global_dir=tmp_path)
    called = []

    def send(p, key, messages, json_mode, timeout):
        called.append(p.base_url)
        return Reply("ok")

    Chain(cfg.chain_providers(private_only=True), send=send).complete([], ident)
    assert called and all(url.startswith("http://localhost") for url in called)
