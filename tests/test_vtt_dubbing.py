import json
from pathlib import Path
import threading
from datetime import datetime

import numpy as np
import pytest
import soundfile as sf

from apps import vtt_dubbing


class DummyTTSEngine:
    def __init__(self, durations: dict[Path, float], raw_duration: float):
        self._durations = durations
        self._raw_duration = raw_duration
        self.last_infer_kwargs = None
        self.last_infer_text = None

    def infer(self, *args, **kwargs):
        self.last_infer_text = args[0] if args else None
        self.last_infer_kwargs = kwargs
        return np.zeros(32, dtype=np.float32)

    def save(self, audio, output_path):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"audio")
        self._durations[output] = self._raw_duration


def install_fake_audio_ops(monkeypatch, durations):
    speed_calls = []
    truncate_calls = []
    padding_calls = []

    def fake_get_audio_duration(audio_path):
        return durations[Path(audio_path)]

    def fake_apply_speed_with_ffmpeg(ffmpeg_path, input_path, output_path, speed, logger, sample_rate=vtt_dubbing.DEFAULT_SAMPLE_RATE):
        source = Path(input_path)
        target = Path(output_path)
        target.write_bytes(source.read_bytes())
        durations[target] = durations[source] / speed
        speed_calls.append(speed)

    def fake_truncate_audio_with_ffmpeg(ffmpeg_path, input_path, output_path, max_duration, logger, sample_rate=vtt_dubbing.DEFAULT_SAMPLE_RATE):
        source = Path(input_path)
        target = Path(output_path)
        target.write_bytes(source.read_bytes())
        durations[target] = min(durations[source], max_duration)
        truncate_calls.append(max_duration)

    def fake_finalize_audio_file(source_path, target_path):
        source = Path(source_path)
        target = Path(target_path)
        target.write_bytes(source.read_bytes())
        durations[target] = durations[source]

    def fake_pad_audio_with_silence(input_path, output_path, target_duration, logger, sample_rate=vtt_dubbing.DEFAULT_SAMPLE_RATE):
        source = Path(input_path)
        target = Path(output_path)
        target.write_bytes(source.read_bytes())
        durations[target] = max(durations[source], target_duration)
        padding_calls.append(target_duration)

    monkeypatch.setattr(vtt_dubbing, "get_audio_duration", fake_get_audio_duration)
    monkeypatch.setattr(vtt_dubbing, "apply_speed_with_ffmpeg", fake_apply_speed_with_ffmpeg)
    monkeypatch.setattr(vtt_dubbing, "truncate_audio_with_ffmpeg", fake_truncate_audio_with_ffmpeg)
    monkeypatch.setattr(vtt_dubbing, "finalize_audio_file", fake_finalize_audio_file)
    monkeypatch.setattr(vtt_dubbing, "pad_audio_with_silence", fake_pad_audio_with_silence)
    return speed_calls, truncate_calls, padding_calls


def make_assignment(speed=1.0):
    return vtt_dubbing.SpeakerAssignment(
        speaker="Narrator",
        voice_data={"codes": np.zeros(4, dtype=np.float32), "text": "ref"},
        source_label="preset:test",
        source_signature="signature",
        speed=speed,
    )


def make_cue(target_duration: float, text: str = "Xin chao ban nhe"):
    return vtt_dubbing.VTTCue(
        index=1,
        speaker="Narrator",
        text=text,
        start_seconds=0.0,
        end_seconds=target_duration,
    )


def make_logger(tmp_path):
    return vtt_dubbing.RunLogger(tmp_path / "vtt.log", reset=True)


def write_test_wav(path: Path, samples: np.ndarray, sample_rate: int = vtt_dubbing.DEFAULT_SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(samples, dtype=np.float32), sample_rate)


def test_build_vtt_output_filename_uses_vtt_stem_and_timestamp():
    filename = vtt_dubbing.build_vtt_output_filename(
        "input/My Demo File.vtt",
        generated_at=datetime(2026, 5, 20, 13, 14, 15),
    )

    assert filename == "My_Demo_File_20260520_131415.wav"


def test_build_vtt_video_output_filename_uses_same_stem_and_timestamp():
    filename = vtt_dubbing.build_vtt_video_output_filename(
        "input/My Demo File.vtt",
        generated_at=datetime(2026, 5, 20, 13, 14, 15),
    )

    assert filename == "My_Demo_File_20260520_131415.mp4"


def test_validate_subtitle_export_mode_defaults_to_none():
    assert vtt_dubbing.validate_subtitle_export_mode(None) == vtt_dubbing.SUBTITLE_EXPORT_NONE


def test_rewrite_problematic_short_cue_maps_ok_to_ok_roi():
    plan = vtt_dubbing.rewrite_problematic_short_cue("ok.")

    assert plan.applied is True
    assert plan.rewritten_text == "ok rồi."
    assert plan.reason == "ok_only"


def test_rewrite_problematic_short_cue_maps_filler_ok_mix_to_a_vang():
    plan = vtt_dubbing.rewrite_problematic_short_cue("À ok")

    assert plan.applied is True
    assert plan.rewritten_text == "à vâng"
    assert plan.reason == "filler_led_mixed_short_phrase"


def test_burn_vtt_subtitles_into_video_uses_subtitles_filter(monkeypatch, tmp_path):
    captured = {}
    input_video = tmp_path / "source.mp4"
    subtitle_path = tmp_path / "subtitle.vtt"
    output_path = tmp_path / "burned.mp4"
    input_video.write_bytes(b"video")
    subtitle_path.write_text("WEBVTT", encoding="utf-8")

    def fake_run_command(command, logger, error_context):
        captured["command"] = command

    monkeypatch.setattr(vtt_dubbing, "run_command", fake_run_command)

    result = vtt_dubbing.burn_vtt_subtitles_into_video(
        ffmpeg_path="ffmpeg",
        input_video_path=input_video,
        vtt_path=subtitle_path,
        output_path=output_path,
        logger=make_logger(tmp_path),
    )

    command = captured["command"]
    assert result == output_path
    assert "-vf" in command
    assert "subtitles=filename=" in command[command.index("-vf") + 1]
    assert "-c:v" in command
    assert "libx264" in command
    assert command[-1] == str(output_path)


def test_add_vtt_subtitle_track_to_video_adds_mov_text_track(monkeypatch, tmp_path):
    captured = {}
    input_video = tmp_path / "source.mp4"
    subtitle_path = tmp_path / "subtitle.vtt"
    output_path = tmp_path / "soft.mp4"
    input_video.write_bytes(b"video")
    subtitle_path.write_text("WEBVTT", encoding="utf-8")

    monkeypatch.setattr(vtt_dubbing, "probe_subtitle_stream_count", lambda *args, **kwargs: 1)

    def fake_run_command(command, logger, error_context):
        captured["command"] = command

    monkeypatch.setattr(vtt_dubbing, "run_command", fake_run_command)

    result = vtt_dubbing.add_vtt_subtitle_track_to_video(
        ffmpeg_path="ffmpeg",
        input_video_path=input_video,
        vtt_path=subtitle_path,
        output_path=output_path,
        logger=make_logger(tmp_path),
        ffprobe_path="ffprobe",
    )

    command = captured["command"]
    assert result == output_path
    assert "1:s:0" in command
    assert "-c:s:1" in command
    assert "mov_text" in command
    assert "title=Vietnamese Subtitle" in command
    assert command[-1] == str(output_path)


def test_mux_dubbed_audio_into_video_appends_default_dubbed_track(monkeypatch, tmp_path):
    captured = {}
    video_path = tmp_path / "source.mp4"
    audio_path = tmp_path / "dubbed.wav"
    output_path = tmp_path / "muxed.mp4"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(vtt_dubbing, "probe_audio_stream_count", lambda *args, **kwargs: 1)

    def fake_run_command(command, logger, error_context):
        captured["command"] = command

    monkeypatch.setattr(vtt_dubbing, "run_command", fake_run_command)

    result = vtt_dubbing.mux_dubbed_audio_into_video(
        ffmpeg_path="ffmpeg",
        video_input_path=video_path,
        dubbed_audio_path=audio_path,
        output_path=output_path,
        logger=make_logger(tmp_path),
        ffprobe_path="ffprobe",
    )

    command = captured["command"]
    assert result == output_path
    assert "-map" in command
    assert "0:a:0" in command
    assert "-disposition:a:1" in command
    assert "default" in command
    assert "-c:a:1" in command
    assert "aac" in command
    assert command[-1] == str(output_path)


def test_cleanup_vtt_artifacts_removes_temp_and_keeps_final_outputs(tmp_path):
    output_root = tmp_path / "outputs"
    temp_audio = output_root / "temp" / "audio"
    logs_dir = output_root / "logs"
    temp_audio.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cue_audio = temp_audio / "cue_000001.wav"
    cue_meta = temp_audio / "cue_000001.json"
    log_file = logs_dir / "vtt_generation.log"
    speaker_file = output_root / "speakers_detected.json"
    final_audio = output_root / "demo_20260520_131415.wav"
    final_video = output_root / "demo_20260520_131415.mp4"

    cue_audio.write_bytes(b"cue")
    cue_meta.write_text("{}", encoding="utf-8")
    log_file.write_text("log", encoding="utf-8")
    speaker_file.write_text("{}", encoding="utf-8")
    final_audio.write_bytes(b"audio")
    final_video.write_bytes(b"video")

    vtt_dubbing.cleanup_vtt_artifacts(
        output_root=output_root,
        logger=make_logger(tmp_path),
        keep_paths=[final_audio, final_video],
    )

    assert final_audio.exists()
    assert final_video.exists()
    assert not (output_root / "temp").exists()
    assert not (output_root / "logs").exists()
    assert not speaker_file.exists()


def test_resolve_log_file_path_uses_shared_default_and_local_custom(monkeypatch, tmp_path):
    shared_log = tmp_path / "shared_logs" / "vtt_generation.log"
    monkeypatch.setattr(vtt_dubbing, "LOG_FILE_PATH", shared_log)

    assert vtt_dubbing.resolve_log_file_path(vtt_dubbing.OUTPUT_ROOT) == shared_log
    assert vtt_dubbing.resolve_log_file_path(tmp_path / "job") == tmp_path / "job" / "logs" / "vtt_generation.log"


def test_apply_auto_overlap_avoidance_keeps_original_starts_when_disabled(tmp_path):
    rendered_cues = [
        vtt_dubbing.CueRenderResult(
            cue=vtt_dubbing.VTTCue(1, "A", "Xin chao", 0.0, 0.5),
            audio_path=tmp_path / "cue1.wav",
            duration_seconds=1.0,
        ),
        vtt_dubbing.CueRenderResult(
            cue=vtt_dubbing.VTTCue(2, "B", "Toi day", 0.5, 1.0),
            audio_path=tmp_path / "cue2.wav",
            duration_seconds=0.8,
        ),
    ]

    adjusted = vtt_dubbing.apply_auto_overlap_avoidance(
        rendered_cues,
        auto_avoid_overlap=False,
        overlap_safety_margin_ms=100.0,
        max_shift_ms=None,
        logger=make_logger(tmp_path),
    )

    assert [item.placement_start_seconds for item in adjusted] == pytest.approx([0.0, 0.5])
    assert [item.shift_applied_ms for item in adjusted] == pytest.approx([0.0, 0.0])
    assert [item.overlap_detected for item in adjusted] == [False, False]


def test_apply_auto_overlap_avoidance_shifts_next_cue_by_overlap_plus_margin(tmp_path):
    rendered_cues = [
        vtt_dubbing.CueRenderResult(
            cue=vtt_dubbing.VTTCue(1, "A", "Xin chao", 0.0, 0.5),
            audio_path=tmp_path / "cue1.wav",
            duration_seconds=1.0,
        ),
        vtt_dubbing.CueRenderResult(
            cue=vtt_dubbing.VTTCue(2, "B", "Toi day", 0.5, 1.0),
            audio_path=tmp_path / "cue2.wav",
            duration_seconds=0.8,
        ),
    ]

    adjusted = vtt_dubbing.apply_auto_overlap_avoidance(
        rendered_cues,
        auto_avoid_overlap=True,
        overlap_safety_margin_ms=100.0,
        max_shift_ms=None,
        logger=make_logger(tmp_path),
    )

    assert [item.placement_start_seconds for item in adjusted] == pytest.approx([0.0, 1.1])
    assert adjusted[1].overlap_detected is True
    assert adjusted[1].shift_applied_ms == pytest.approx(600.0)
    assert adjusted[1].auto_avoid_overlap_enabled is True


def test_apply_auto_overlap_avoidance_respects_max_shift_ms_cap(tmp_path):
    rendered_cues = [
        vtt_dubbing.CueRenderResult(
            cue=vtt_dubbing.VTTCue(1, "A", "Xin chao", 0.0, 0.5),
            audio_path=tmp_path / "cue1.wav",
            duration_seconds=1.0,
        ),
        vtt_dubbing.CueRenderResult(
            cue=vtt_dubbing.VTTCue(2, "B", "Toi day", 0.5, 1.0),
            audio_path=tmp_path / "cue2.wav",
            duration_seconds=0.8,
        ),
    ]

    adjusted = vtt_dubbing.apply_auto_overlap_avoidance(
        rendered_cues,
        auto_avoid_overlap=True,
        overlap_safety_margin_ms=100.0,
        max_shift_ms=250.0,
        logger=make_logger(tmp_path),
    )

    assert [item.placement_start_seconds for item in adjusted] == pytest.approx([0.0, 0.75])
    assert adjusted[1].shift_applied_ms == pytest.approx(250.0)
    assert adjusted[1].overlap_detected is True


def test_run_command_logs_oserror(monkeypatch, tmp_path):
    logger = make_logger(tmp_path)

    def fake_subprocess_run(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(vtt_dubbing.subprocess, "run", fake_subprocess_run)

    with pytest.raises(vtt_dubbing.VTTDubbingError, match="spawn failed"):
        vtt_dubbing.run_command(["ffmpeg", "-version"], logger, "FFmpeg test lỗi")

    log_text = (tmp_path / "vtt.log").read_text(encoding="utf-8")
    assert "FFmpeg test lỗi: spawn failed" in log_text


def test_render_single_cue_passes_target_duration_to_tts(monkeypatch, tmp_path):
    durations = {}
    install_fake_audio_ops(monkeypatch, durations)
    engine = DummyTTSEngine(durations, raw_duration=0.6)
    cue = make_cue(1.5, text="ok")
    assignment = make_assignment(speed=1.0)

    result = vtt_dubbing.render_single_cue(
        cue=cue,
        assignment=assignment,
        tts_engine=engine,
        tts_lock=threading.Lock(),
        ffmpeg_path="ffmpeg",
        temp_audio_dir=tmp_path / "temp",
        selected_mode="standard",
        emotion="natural",
        overflow_mode="keep",
        max_speedup=1.25,
        logger=make_logger(tmp_path),
    )

    assert result.audio_path is not None
    assert engine.last_infer_kwargs["target_duration"] == pytest.approx(1.5)


def test_render_single_cue_rewrites_problematic_short_text_before_tts(monkeypatch, tmp_path):
    durations = {}
    install_fake_audio_ops(monkeypatch, durations)
    engine = DummyTTSEngine(durations, raw_duration=0.6)
    cue = make_cue(1.5, text="À ok")
    assignment = make_assignment(speed=1.0)

    result = vtt_dubbing.render_single_cue(
        cue=cue,
        assignment=assignment,
        tts_engine=engine,
        tts_lock=threading.Lock(),
        ffmpeg_path="ffmpeg",
        temp_audio_dir=tmp_path / "temp",
        selected_mode="standard",
        emotion="natural",
        overflow_mode="keep",
        max_speedup=1.25,
        logger=make_logger(tmp_path),
    )

    assert result.audio_path is not None
    assert engine.last_infer_text == "à vâng"


def test_compose_with_ffmpeg_falls_back_to_internal_mixer_for_large_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(vtt_dubbing, "FFMPEG_TIMELINE_COMPOSE_MAX_INPUTS", 1)

    def fail_run_command(*args, **kwargs):
        raise AssertionError("FFmpeg filtergraph path should not be used for this test")

    monkeypatch.setattr(vtt_dubbing, "run_command", fail_run_command)

    cue_1_path = tmp_path / "cue_1.wav"
    cue_2_path = tmp_path / "cue_2.wav"
    write_test_wav(cue_1_path, np.full(2400, 0.25, dtype=np.float32))
    write_test_wav(cue_2_path, np.full(2400, 0.5, dtype=np.float32))

    rendered_cues = [
        vtt_dubbing.CueRenderResult(
            cue=vtt_dubbing.VTTCue(
                index=1,
                speaker="Narrator",
                text="One",
                start_seconds=0.0,
                end_seconds=0.1,
            ),
            audio_path=cue_1_path,
            duration_seconds=0.1,
        ),
        vtt_dubbing.CueRenderResult(
            cue=vtt_dubbing.VTTCue(
                index=2,
                speaker="Narrator",
                text="Two",
                start_seconds=0.5,
                end_seconds=0.6,
            ),
            audio_path=cue_2_path,
            duration_seconds=0.1,
        ),
    ]

    output_path = tmp_path / "mixed.wav"
    result = vtt_dubbing.compose_with_ffmpeg(
        rendered_cues,
        output_path,
        "ffmpeg",
        make_logger(tmp_path),
    )

    mixed_audio, sample_rate = sf.read(str(result), dtype="float32")
    assert result == output_path
    assert sample_rate == vtt_dubbing.DEFAULT_SAMPLE_RATE
    assert mixed_audio.shape[0] >= int(0.6 * sample_rate)
    assert mixed_audio[0] == pytest.approx(0.25, abs=2e-4)
    assert mixed_audio[int(0.5 * sample_rate)] == pytest.approx(0.5, abs=2e-4)
    assert not (tmp_path / "temp" / "mixed__timeline.f32").exists()


def test_render_single_cue_auto_fits_long_audio(monkeypatch, tmp_path):
    durations = {}
    speed_calls, truncate_calls, padding_calls = install_fake_audio_ops(monkeypatch, durations)
    engine = DummyTTSEngine(durations, raw_duration=4.0)

    result = vtt_dubbing.render_single_cue(
        cue=make_cue(2.0),
        assignment=make_assignment(speed=1.0),
        tts_engine=engine,
        tts_lock=threading.Lock(),
        ffmpeg_path="ffmpeg",
        temp_audio_dir=tmp_path,
        selected_mode="standard",
        emotion="natural",
        overflow_mode="keep",
        max_speedup=3.0,
        logger=make_logger(tmp_path),
    )

    assert result.duration_seconds == pytest.approx(2.0)
    assert speed_calls == pytest.approx([2.0])
    assert truncate_calls == []
    assert padding_calls == []


def test_render_single_cue_applies_base_speed_before_auto_fit(monkeypatch, tmp_path):
    durations = {}
    speed_calls, truncate_calls, padding_calls = install_fake_audio_ops(monkeypatch, durations)
    engine = DummyTTSEngine(durations, raw_duration=4.0)

    result = vtt_dubbing.render_single_cue(
        cue=make_cue(2.0),
        assignment=make_assignment(speed=1.25),
        tts_engine=engine,
        tts_lock=threading.Lock(),
        ffmpeg_path="ffmpeg",
        temp_audio_dir=tmp_path,
        selected_mode="standard",
        emotion="natural",
        overflow_mode="keep",
        max_speedup=3.0,
        logger=make_logger(tmp_path),
    )

    assert result.duration_seconds == pytest.approx(2.0)
    assert speed_calls == pytest.approx([1.25, 1.6])
    assert truncate_calls == []
    assert padding_calls == []


def test_render_single_cue_auto_slows_down_non_protected_audio(monkeypatch, tmp_path):
    durations = {}
    speed_calls, truncate_calls, padding_calls = install_fake_audio_ops(monkeypatch, durations)
    engine = DummyTTSEngine(durations, raw_duration=1.0)

    result = vtt_dubbing.render_single_cue(
        cue=make_cue(2.0, text="Xin chao cac ban nhe"),
        assignment=make_assignment(speed=1.0),
        tts_engine=engine,
        tts_lock=threading.Lock(),
        ffmpeg_path="ffmpeg",
        temp_audio_dir=tmp_path,
        selected_mode="standard",
        emotion="natural",
        overflow_mode="keep",
        max_speedup=2.0,
        logger=make_logger(tmp_path),
    )

    assert result.duration_seconds == pytest.approx(2.0)
    assert speed_calls == pytest.approx([0.5])
    assert truncate_calls == []
    assert padding_calls == []


def test_render_single_cue_uses_overflow_speedup_after_natural_cap(monkeypatch, tmp_path):
    durations = {}
    speed_calls, truncate_calls, padding_calls = install_fake_audio_ops(monkeypatch, durations)
    engine = DummyTTSEngine(durations, raw_duration=4.0)

    result = vtt_dubbing.render_single_cue(
        cue=make_cue(1.0),
        assignment=make_assignment(speed=1.0),
        tts_engine=engine,
        tts_lock=threading.Lock(),
        ffmpeg_path="ffmpeg",
        temp_audio_dir=tmp_path,
        selected_mode="standard",
        emotion="natural",
        overflow_mode="speedup",
        max_speedup=1.25,
        logger=make_logger(tmp_path),
    )

    assert result.duration_seconds == pytest.approx(1.0)
    assert speed_calls == pytest.approx([1.25, 3.2])
    assert truncate_calls == []
    assert padding_calls == []


def test_render_single_cue_protected_short_text_uses_silence_padding_instead_of_slowdown(monkeypatch, tmp_path):
    durations = {}
    speed_calls, truncate_calls, padding_calls = install_fake_audio_ops(monkeypatch, durations)
    engine = DummyTTSEngine(durations, raw_duration=1.0)

    result = vtt_dubbing.render_single_cue(
        cue=make_cue(2.0, text="Vâng!"),
        assignment=make_assignment(speed=1.0),
        tts_engine=engine,
        tts_lock=threading.Lock(),
        ffmpeg_path="ffmpeg",
        temp_audio_dir=tmp_path,
        selected_mode="standard",
        emotion="natural",
        overflow_mode="keep",
        max_speedup=2.0,
        logger=make_logger(tmp_path),
    )

    meta_path = tmp_path / "cue_000001__Narrator.json"
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))

    assert result.duration_seconds == pytest.approx(2.0)
    assert speed_calls == []
    assert truncate_calls == []
    assert padding_calls == pytest.approx([2.0])
    assert meta_payload["cue_timing"]["protected_short_text"] is True
    assert meta_payload["cue_timing"]["silence_padding_seconds"] == pytest.approx(1.0)
    assert "silence_padding" in meta_payload["cue_timing"]["timing_actions"]


def test_render_single_cue_protected_short_text_allows_speedup_when_too_long(monkeypatch, tmp_path):
    durations = {}
    speed_calls, truncate_calls, padding_calls = install_fake_audio_ops(monkeypatch, durations)
    engine = DummyTTSEngine(durations, raw_duration=2.0)

    result = vtt_dubbing.render_single_cue(
        cue=make_cue(1.0, text="dạ"),
        assignment=make_assignment(speed=1.0),
        tts_engine=engine,
        tts_lock=threading.Lock(),
        ffmpeg_path="ffmpeg",
        temp_audio_dir=tmp_path,
        selected_mode="standard",
        emotion="natural",
        overflow_mode="keep",
        max_speedup=3.0,
        logger=make_logger(tmp_path),
    )

    assert result.duration_seconds == pytest.approx(1.0)
    assert speed_calls == pytest.approx([2.0])
    assert truncate_calls == []
    assert padding_calls == []


def test_render_single_cue_protected_short_text_clamps_manual_base_slowdown(monkeypatch, tmp_path):
    durations = {}
    speed_calls, truncate_calls, padding_calls = install_fake_audio_ops(monkeypatch, durations)
    engine = DummyTTSEngine(durations, raw_duration=1.0)

    result = vtt_dubbing.render_single_cue(
        cue=make_cue(2.0, text="ừm"),
        assignment=make_assignment(speed=0.8),
        tts_engine=engine,
        tts_lock=threading.Lock(),
        ffmpeg_path="ffmpeg",
        temp_audio_dir=tmp_path,
        selected_mode="standard",
        emotion="natural",
        overflow_mode="keep",
        max_speedup=2.0,
        logger=make_logger(tmp_path),
    )

    meta_path = tmp_path / "cue_000001__Narrator.json"
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))

    assert result.duration_seconds == pytest.approx(2.0)
    assert speed_calls == []
    assert truncate_calls == []
    assert padding_calls == pytest.approx([2.0])
    assert meta_payload["cue_timing"]["effective_base_speed"] == pytest.approx(1.0)
    assert meta_payload["cue_timing"]["final_speed"] == pytest.approx(1.0)
