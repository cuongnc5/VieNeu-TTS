import logging

import numpy as np

from vieneu.base import BaseVieneuTTS
from vieneu.logging_utils import configure_file_logger
from vieneu.utils import extract_speech_ids


class DummyTTS(BaseVieneuTTS):
    def infer(self, text: str, apply_watermark: bool = True, **kwargs):
        return np.array([], dtype=np.float32)

    def infer_batch(self, texts, apply_watermark: bool = True, **kwargs):
        return []


def test_generation_diagnostics_warn_on_possible_elongation(monkeypatch, caplog):
    monkeypatch.setenv("VIENEU_DIAGNOSTIC_LOGGING", "1")
    tts = DummyTTS()
    test_logger = logging.getLogger("Vieneu.Test")
    output_str = "".join("<|speech_7|>" for _ in range(90))

    with caplog.at_level(logging.INFO, logger="Vieneu.Test"):
        tts._log_generation_result(
            test_logger,
            chunk_index=0,
            total_chunks=1,
            chunk_text="xin",
            chunk_phonemes="x-in",
            output_str=output_str,
            wav=np.zeros(90 * tts.hop_length, dtype=np.float32),
        )

    assert "[diag][warn]" in caplog.text
    assert "possible_elongation" in caplog.text
    assert "repeat_run=90" in caplog.text


def test_configure_file_logger_writes_to_explicit_path(tmp_path):
    log_path = tmp_path / "dubbing_diagnostics.log"
    logger = configure_file_logger("Vieneu.TestFile", log_path=log_path)

    logger.info("diagnostic line")
    for handler in logger.handlers:
        handler.flush()

    assert log_path.exists()
    assert "diagnostic line" in log_path.read_text(encoding="utf-8")


def test_generation_guard_trims_pathological_short_cue():
    tts = DummyTTS()
    test_logger = logging.getLogger("Vieneu.TestGuard")
    guard = tts._build_generation_guard("ok", "o-k")
    output_str = "".join(f"<|speech_{idx}|>" for idx in range(200))

    trimmed = tts._apply_generation_guard(
        test_logger,
        chunk_text="ok",
        chunk_phonemes="o-k",
        output_str=output_str,
        guard=guard,
    )

    assert len(extract_speech_ids(trimmed)) == guard["speech_token_cap"]


def test_generation_guard_marks_repetitive_compact_phrase_as_short():
    tts = DummyTTS()

    guard = tts._build_generation_guard("A, ok ok ok.", "a ok ok ok")

    assert guard["short_text"] is True
    assert guard["repetitive_compact"] is True
    assert guard["max_new_tokens"] < 150


def test_prepare_text_for_inference_rewrites_problematic_short_text(caplog):
    tts = DummyTTS()
    test_logger = logging.getLogger("Vieneu.TestRewrite")

    with caplog.at_level(logging.INFO, logger="Vieneu.TestRewrite"):
        rewritten, plan = tts._prepare_text_for_inference(test_logger, "À ok")

    assert rewritten == "à vâng"
    assert plan.applied is True
    assert plan.reason == "filler_led_mixed_short_phrase"
    assert "Rewrite problematic short text for stable synthesis" in caplog.text
