import io
import wave

import numpy as np
import pytest

from voxtex.audio import capture, stt
from voxtex.audio.capture import FRAME_MS, FRAME_SAMPLES, Segmenter, pick_device
from voxtex.audio.turns import TurnTimer, is_filler
from voxtex.config import load_config

LOUD = np.full(FRAME_SAMPLES, 3000, dtype=np.int16)
QUIET = np.zeros(FRAME_SAMPLES, dtype=np.int16)


def feed(seg, pattern):
    """pattern: list of (frames, loud?) runs. Returns utterances produced."""
    out, t = [], 0.0
    for n, loud in pattern:
        for _ in range(n):
            t += FRAME_MS / 1000
            u = seg.feed(LOUD if loud else QUIET, t)
            if u is not None:
                out.append(u)
    return out


def energy_vad(frame: bytes) -> bool:
    return np.frombuffer(frame, dtype=np.int16).any()


def test_segmenter_cuts_on_pause_with_pre_roll():
    seg = Segmenter(energy_vad, pause_s=0.7)
    us = feed(seg, [(20, False), (34, True), (30, False)])  # ~1 s speech, 0.9 s silence
    assert len(us) == 1
    u = us[0]
    # pre-roll (5 silent + the 5 voiced frames that started it) + 29 more voiced
    # + 23 silent frames until the 0.7 s pause
    assert len(u.audio) == (10 + 29 + 23) * FRAME_SAMPLES
    assert u.audio.dtype == np.float32 and u.audio.max() == pytest.approx(3000 / 32768)
    assert u.ended == pytest.approx(54 * FRAME_MS / 1000)


def test_segmenter_ignores_blips_and_keeps_short_pauses_inside():
    seg = Segmenter(energy_vad, pause_s=0.7, min_speech_s=0.3)
    assert feed(seg, [(2, True), (30, False)]) == []  # too short to start
    assert feed(seg, [(4, True), (30, False)]) == []  # starts, but < 0.3 s voiced
    us = feed(seg, [(15, True), (10, False), (15, True), (30, False)])  # 0.3 s gap: same utterance
    assert len(us) == 1


def test_segmenter_max_length():
    seg = Segmenter(energy_vad, pause_s=0.7, max_utterance_s=1.0)
    us = feed(seg, [(100, True)])  # 3 s without a pause
    assert len(us) == 3  # cut about every second
    assert all(u.seconds <= 1.0 for u in us)


@pytest.mark.parametrize(
    "text, filler",
    [
        ("Um.", True), ("εμμ...", True), ("Hmm, let me think", False), ("let me think", True),
        ("Ευχαριστώ.", True), ("Υπότιτλοι AUTHORWAVE", True), ("", True), ("...", True),
        ("We use two encoders.", False), ("εμ, το RDF", False),
    ],
)  # fmt: skip
def test_is_filler(text, filler):
    assert is_filler(text) is filler


def test_prompt_echo_is_dropped():
    prompt = "Διπλωματική εργασία. Chalkidis, Broekstra"
    assert is_filler("Chalkidis, Broekstra", prompt)
    assert is_filler("Διπλωματική εργασία.", prompt)
    assert not is_filler("Ο Broekstra έδειξε ότι", prompt)


def test_turn_timer():
    t = TurnTimer(decision_pause=2.5, max_pending_s=20)
    long_sentence = "We fine-tune each encoder for three epochs."
    assert t.check("", None, 0, False, 10) == (False, False)
    # a finished sentence after a breath: maybe more is coming, so not yet
    assert t.check(long_sentence, 1.0, 9.6, False, 10) == (False, False)
    # ...but after a clear pause, decide (not forced)
    assert t.check(long_sentence, 1.0, 8.7, False, 10) == (True, False)
    # fragment: wait for the long pause, then force
    assert t.check("so we use", 1.0, 8.0, False, 10) == (False, False)
    assert t.check("so we use", 1.0, 7.0, False, 10) == (True, True)
    # still talking: never, however long it has been
    assert t.check(long_sentence, 1.0, 9.9, True, 10) == (False, False)
    assert t.check(long_sentence, 1.0, 9.9, True, 22) == (False, False)
    assert t.check(long_sentence, 1.0, 21.9, False, 22) == (True, True)  # the next pause
    # after the model said "wait", the same text isn't sent again until the pause
    t.model_waited(long_sentence)
    assert t.check(long_sentence, 1.0, 8.7, False, 10) == (False, False)
    assert t.check(long_sentence, 1.0, 7.0, False, 10) == (True, True)


@pytest.mark.parametrize(
    "gpu, key, private, mode, expected",
    [
        (True, "k", False, "auto", ("local", "large-v3-turbo", "cuda")),
        (False, "k", False, "auto", ("groq", "whisper-large-v3-turbo", "cpu")),
        (False, None, False, "auto", ("local", "small", "cpu")),
        (False, "k", True, "auto", ("local", "small", "cpu")),
        (True, "k", False, "groq", ("groq", "whisper-large-v3-turbo", "cpu")),
        (False, "k", False, "local", ("local", "small", "cpu")),
    ],
)
def test_choose_stt(monkeypatch, tmp_path, gpu, key, private, mode, expected):
    monkeypatch.setattr(stt, "_cuda", gpu)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    if key:
        monkeypatch.setenv("GROQ_API_KEY", key)
    cfg = load_config(None, global_dir=tmp_path)
    c = stt.choose_stt(cfg, private, mode)
    assert (c.kind, c.model, c.device) == expected
    assert c.reason


def test_english_only_models_are_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(stt, "_cuda", False)
    cfg = load_config(None, global_dir=tmp_path)
    cfg.stt_local_model = "small.en"
    assert stt.choose_stt(cfg, private=True).model == "small"


def test_to_wav():
    audio = np.linspace(-1, 1, 16000).astype(np.float32)
    with wave.open(io.BytesIO(stt.to_wav(audio))) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 16000)


def test_cloud_whisper_request(monkeypatch):
    sent = {}

    class Fake:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kw):
                    sent.update(kw)
                    return type("R", (), {"text": " Το RDF. "})()

    w = stt.CloudWhisper("whisper-large-v3-turbo", "https://x", "k", language="el")
    w.client = Fake
    r = w.transcribe(np.zeros(8000, dtype=np.float32), "Διπλωματική εργασία. Broekstra")
    assert r.text == "Το RDF." and r.audio_seconds == 0.5
    assert sent["language"] == "el" and sent["prompt"].endswith("Broekstra")
    assert sent["file"][0] == "utterance.wav"


def test_pick_device(monkeypatch):
    monkeypatch.setattr(
        capture, "list_inputs", lambda: [(0, "default"), (3, "pulse"), (5, "USB Mic")]
    )
    assert pick_device("") == 3  # WSL: prefer the PulseAudio bridge
    assert pick_device("usb") == 5
    assert pick_device("5") == 5
    with pytest.raises(capture.MicError):
        pick_device("bluetooth")
    monkeypatch.setattr(capture, "list_inputs", lambda: [])
    with pytest.raises(capture.MicError):
        pick_device("")


def test_speech_to_text_falls_back(monkeypatch):
    from voxtex.audio.capture import Utterance
    from voxtex.listenmode import SpeechToText

    class Broken:
        name = "groq x"

        def transcribe(self, audio, prompt=""):
            raise RuntimeError("429")

    class Local:
        name = "local small"

        def transcribe(self, audio, prompt=""):
            return stt.Transcript("hello", 0.1, 1.0, self.name)

    s = SpeechToText(Broken(), "", Local())
    s.start()
    s.inbox.put(Utterance(np.zeros(16000, dtype=np.float32), 0, 1))
    heard = s.outbox.get(timeout=5)
    assert heard.text == "hello" and "switched to local small" in s.notice


def test_whisper_prompt(sample_thesis):
    from voxtex.listenmode import whisper_prompt
    from voxtex.project.index import build_index

    assert whisper_prompt(build_index(sample_thesis)) == (
        "Thesis notes. Chalkidis, Devlin, Vaswani, Zheng"
    )


def test_spoken_terms_filter():
    from pathlib import Path

    from voxtex.project.discover import SourceFile
    from voxtex.project.index import _is_spoken_term, _wrapped_terms

    for term in ("RDF", "SeRQL", "RDF Schema", "Peer-to-Peer systems", "Berners-Lee"):
        assert _is_spoken_term(term), term
    for term in ("http://w3.org/x", "classesToWrite", "peer1", "#2", "a b c d", "x" * 30):
        assert not _is_spoken_term(term), term
    text = "\\en{RDF} \\en{RDF} \\en{rdf} \\en{Sesame} \\en{http://x.org} \\en{groupedMapping}"
    assert _wrapped_terms([SourceFile(Path("c.tex"), text)], "\\en") == ["RDF", "Sesame"]


def test_stt_languages_and_greek_hint(tmp_path):
    from voxtex.listenmode import stt_languages
    from voxtex.project.index import build_index

    root = tmp_path / "gr"
    root.mkdir()
    (root / "main.tex").write_text(
        "\\documentclass{report}\\usepackage[english,greek]{babel}"
        "\\newcommand{\\en}[1]{\\foreignlanguage{english}{#1}}\\begin{document}\\input{ch}"
        "\\end{document}"
    )
    (root / "ch.tex").write_text("Το \\en{RDF} και το \\en{RDF} και το \\en{Sesame}.")
    idx = build_index(root)
    assert stt_languages(idx) == ("el", "en")
    from voxtex.listenmode import whisper_prompt

    assert whisper_prompt(idx) == "Διπλωματική εργασία. RDF, Sesame"


def test_language_choice_is_limited_to_allowed():
    w = stt.LocalWhisper("x", languages=("el", "en"))

    class FakeCT2:
        @staticmethod
        def detect_language(enc):
            return [[("<|zh|>", 0.5), ("<|ru|>", 0.3), ("<|el|>", 0.15), ("<|en|>", 0.05)]]

    w._model = type("M", (), {"model": FakeCT2})()
    assert w._pick_language(None, 3.0) == "el"


def test_language_switch_needs_clear_evidence():
    w = stt.LocalWhisper("x", languages=("el", "en"))
    assert w.last_language == "el"  # the document's language to start with
    assert w._sticky([("en", 0.9), ("el", 0.1)], 0.8) == "el"  # too short to trust
    assert w._sticky([("en", 0.55), ("el", 0.45)], 4.0) == "el"  # too close to call
    assert w._sticky([("en", 0.9), ("el", 0.1)], 3.0) == "en"  # a clear English sentence
    assert w._sticky([("el", 0.6), ("en", 0.4)], 3.0) == "en"  # stays until clear again
    assert w._sticky([("el", 0.95), ("en", 0.05)], 2.0) == "el"


def test_reasoning_effort_sent_and_dropped_if_unsupported(monkeypatch):
    import openai

    from voxtex.config import ProviderConfig
    from voxtex.providers import send_openai

    calls = []

    class Resp:
        headers = {}

        def parse(self):
            msg = type("Msg", (), {"content": '{"action":"wait"}'})
            choice = type("C", (), {"message": msg})
            return type("R", (), {"choices": [choice], "usage": None})()

    class Fake:
        def __init__(self, **kw):
            self.chat = self

        @property
        def completions(self):
            return self

        @property
        def with_raw_response(self):
            return self

        def create(self, **kw):
            calls.append(kw)
            if len(calls) == 1:
                import httpx2 as httpx

                req = httpx.Request("POST", "http://x")
                raise openai.BadRequestError(
                    "reasoning_effort is not supported", response=httpx.Response(400, request=req),
                    body=None,
                )  # fmt: skip
            return Resp()

    monkeypatch.setattr(openai, "OpenAI", Fake)
    p = ProviderConfig(name="g", base_url="http://x", model="m", reasoning_effort="low")
    assert send_openai(p, "k", [], True, 5).text == '{"action":"wait"}'
    assert calls[0]["reasoning_effort"] == "low" and "reasoning_effort" not in calls[1]


def test_match_terms_finds_thesis_spelling_of_misheard_words():
    from voxtex.engine.retrieve import match_terms

    terms = ["SeRQL", "RDF", "RQL", "Sesame", "RDF Schema", "layout", "Jena"]
    assert match_terms("Γράψε SCRQL αντί για RQL", terms) == ["SeRQL", "RQL"]
    assert match_terms("το Jenna και το Sesami", terms) == ["Sesame", "Jena"]
    assert match_terms("τίποτα σχετικό εδώ", terms) == []


def test_noise_gate_learns_the_room():
    from voxtex.audio.capture import FRAME_SAMPLES, NoiseGate

    rng = np.random.default_rng(0)

    def frame(rms):
        return (rng.normal(0, rms, FRAME_SAMPLES) * 32768).astype(np.int16).tobytes()

    gate = NoiseGate()
    fan = [gate.loud_enough(frame(0.01)) for _ in range(400)]  # 12 s of a noisy fan
    assert not any(fan[-100:])  # learned: the fan is not speech
    assert all(gate.loud_enough(frame(0.08)) for _ in range(30))  # talking over it
    quiet = NoiseGate()
    for _ in range(100):
        quiet.loud_enough(frame(0.0003))  # a quiet room
    assert quiet.loud_enough(frame(0.003))  # soft speech still gets through


@pytest.mark.parametrize(
    "text, garbled",
    [
        ("Υπότιτλοι AUTHORWAVE", True),
        ("İzlediğiniz için teşekkür ederim.", True),
        ("All right. All right. All right.", True),
        ("Περδίδι, Περδίδι, Περδίδι", True),
        ("Andes-Lee, 커버 at core 20 minutes.", True),
        ("Το Sesame υποστηρίζει ερωτήματα RQL, κόμμα, και SeRQL.", False),
        ("Create a table, 5 rows and 2 columns.", False),
        ("Σύμφωνα με τον Müller, café.", False),
    ],
)
def test_is_garbled(text, garbled):
    from voxtex.audio.turns import is_garbled

    assert is_garbled(text) is garbled


def test_spoken_undo():
    from voxtex.audio.turns import spoken_undo

    assert spoken_undo("Undo the last change.") == 1
    assert spoken_undo("Undo twice.") == 2
    assert spoken_undo("Αναίρεσε το δύο φορές") == 2
    assert spoken_undo("Undo the changes we made to the table") == 0  # dictation
    assert spoken_undo("Πρέπει να κάνω undo") == 0


class _Features:
    nb_max_frames = 3000

    def __call__(self, audio):
        return np.zeros((80, 100), dtype=np.float32)


class _FakeWhisperModel:
    feature_extractor = _Features()

    def encode(self, features):
        return "encoded"


def test_an_unsure_transcript_is_tried_in_the_other_language(monkeypatch):
    w = stt.LocalWhisper("x", languages=("el", "en"))
    w._model = _FakeWhisperModel()
    heard = {"en": ("Belit.", -1.4, "en"), "el": ("Βρες το.", -0.3, "el")}
    monkeypatch.setattr(w, "_decode", lambda audio, lang, prompt: heard[lang])
    monkeypatch.setattr(w, "_pick_language", lambda encoded, seconds: "en")
    t = w.transcribe(np.zeros(16000, dtype=np.float32))
    assert (t.text, t.language, w.last_language) == ("Βρες το.", "el", "el")
    heard["en"] = ("Undo twice.", -0.2, "en")  # sure enough: no second pass
    assert w.transcribe(np.zeros(16000, dtype=np.float32)).text == "Undo twice."


def test_model_download_reports_progress(monkeypatch):
    import huggingface_hub
    from faster_whisper import utils

    def not_cached(model, local_files_only=False):
        raise OSError("not cached")

    def download(repo, allow_patterns, tqdm_class):  # what huggingface_hub does
        bar = tqdm_class(total=0, unit="B")
        bar.total = 1_600_000_000
        for _ in range(4):
            bar.update(400_000_000)

    monkeypatch.setattr(utils, "download_model", not_cached)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    seen = []
    stt.fetch_model("large-v3-turbo", lambda done, total: seen.append((done, total)))
    assert seen[-1] == (1600.0, 1600.0) and len(seen) == 4


async def test_editor_shows_the_download(sample_thesis, tmp_path):
    from voxtex.config import ProviderConfig, load_config
    from voxtex.editor.app import EditorApp
    from voxtex.project.index import load_index
    from voxtex.providers import Chain

    idx, _ = load_index(sample_thesis)
    config = load_config(idx.root_path, global_dir=tmp_path / "noconfig")
    local = ProviderConfig(name="local", base_url="http://localhost:11434/v1", model="m")
    path = sample_thesis / "chapters" / "methodology.tex"
    app = EditorApp(idx, config, Chain([local], send=lambda *a: None), path, audio=False)
    async with app.run_test():
        app.phase = "loading"
        app.downloading(640, 1600)
        line = app.heard_line(100).plain
        assert "downloading the speech model" in line and "640 / 1,600 MB" in line
