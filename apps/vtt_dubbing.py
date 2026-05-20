from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import threading
import time
import unicodedata

import numpy as np
import soundfile as sf


MAX_VTT_SPEAKERS = 16
DEFAULT_SAMPLE_RATE = 24_000
CUE_TIMING_TOLERANCE_SECONDS = 0.01
VTT_CACHE_SCHEMA_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "vtt"
TEMP_AUDIO_DIR = OUTPUT_ROOT / "temp" / "audio"
LOG_FILE_PATH = OUTPUT_ROOT / "logs" / "vtt_generation.log"
SPEAKER_EXPORT_PATH = OUTPUT_ROOT / "speakers_detected.json"
SUPPORTED_REFERENCE_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SUPPORTED_VIDEO_SUFFIXES = {".mp4"}
SUBTITLE_EXPORT_NONE = "None"
SUBTITLE_EXPORT_BURN = "Burn subtitle into video"
SUBTITLE_EXPORT_SOFT = "Add subtitle track only"
SUPPORTED_SUBTITLE_EXPORT_MODES = {
    SUBTITLE_EXPORT_NONE,
    SUBTITLE_EXPORT_BURN,
    SUBTITLE_EXPORT_SOFT,
}

VOICE_TAG_PATTERN = re.compile(r"^\s*<v\s+([^>]+)>(.*?)</v>\s*$", re.IGNORECASE | re.DOTALL)
BRACKET_SPEAKER_PATTERN = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$", re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")
SPEAKER_SEPARATORS = (":", "：")


class VTTDubbingError(RuntimeError):
    """Raised when VTT dubbing cannot continue."""


@dataclass(frozen=True)
class VTTCue:
    index: int
    speaker: str
    text: str
    start_seconds: float
    end_seconds: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


@dataclass(frozen=True)
class SpeakerAssignment:
    speaker: str
    voice_data: dict[str, Any]
    source_label: str
    source_signature: str
    speed: float


@dataclass(frozen=True)
class CueRenderResult:
    cue: VTTCue
    audio_path: Optional[Path]
    duration_seconds: float
    skipped: bool = False


@dataclass(frozen=True)
class CueTimingPlan:
    target_duration: float
    measured_duration: float
    required_ratio: float
    applied_ratio: float
    min_ratio: float
    max_ratio: float

    @property
    def duration_delta(self) -> float:
        return self.measured_duration - self.target_duration

    @property
    def was_clamped(self) -> bool:
        return abs(self.applied_ratio - self.required_ratio) > 1e-6

    @property
    def needs_adjustment(self) -> bool:
        return (
            abs(self.duration_delta) > CUE_TIMING_TOLERANCE_SECONDS
            and abs(self.applied_ratio - 1.0) > 1e-6
        )


class RunLogger:
    """Thread-safe run logger that writes to disk and keeps a UI-friendly buffer."""

    def __init__(self, file_path: Path, reset: bool = False):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.file_path.write_text("", encoding="utf-8")
        self._messages: list[str] = []
        self._lock = threading.Lock()

    def log(self, level: str, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {level}: {message}"
        with self._lock:
            self._messages.append(line)
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def info(self, message: str) -> None:
        self.log("INFO", message)

    def warning(self, message: str) -> None:
        self.log("WARNING", message)

    def error(self, message: str) -> None:
        self.log("ERROR", message)

    def render(self) -> str:
        with self._lock:
            return "\n".join(self._messages[-400:])


def ensure_output_tree(output_root: Path = OUTPUT_ROOT) -> None:
    (output_root / "temp" / "audio").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)


def append_log_line(message: str, output_root: Path = OUTPUT_ROOT, reset: bool = False) -> None:
    ensure_output_tree(output_root)
    logger = RunLogger(output_root / "logs" / "vtt_generation.log", reset=reset)
    logger.info(message)


def normalize_speaker_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", (name or "").strip())
    return cleaned or "Unknown"


def sanitize_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_only).strip("._")
    return sanitized or "speaker"


def build_vtt_output_basename(vtt_path: str | Path, generated_at: Optional[datetime] = None) -> str:
    timestamp = (generated_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    stem = sanitize_filename(Path(vtt_path).stem) or "vtt"
    return f"{stem}_{timestamp}"


def build_vtt_output_filename(vtt_path: str | Path, generated_at: Optional[datetime] = None) -> str:
    return f"{build_vtt_output_basename(vtt_path, generated_at=generated_at)}.wav"


def build_vtt_output_path(
    vtt_path: str | Path,
    output_root: Path = OUTPUT_ROOT,
    generated_at: Optional[datetime] = None,
) -> Path:
    return output_root / build_vtt_output_filename(vtt_path, generated_at=generated_at)


def build_vtt_video_output_filename(vtt_path: str | Path, generated_at: Optional[datetime] = None) -> str:
    return f"{build_vtt_output_basename(vtt_path, generated_at=generated_at)}.mp4"


def build_vtt_video_output_path(
    vtt_path: str | Path,
    output_root: Path = OUTPUT_ROOT,
    generated_at: Optional[datetime] = None,
) -> Path:
    return output_root / build_vtt_video_output_filename(vtt_path, generated_at=generated_at)


def validate_subtitle_export_mode(subtitle_export_mode: str | None) -> str:
    mode = (subtitle_export_mode or SUBTITLE_EXPORT_NONE).strip() or SUBTITLE_EXPORT_NONE
    if mode not in SUPPORTED_SUBTITLE_EXPORT_MODES:
        supported = ", ".join(sorted(SUPPORTED_SUBTITLE_EXPORT_MODES))
        raise VTTDubbingError(f"Subtitle export mode không hợp lệ. Các giá trị hỗ trợ: {supported}")
    return mode


def build_subtitles_filter(vtt_path: str | Path) -> str:
    path_value = str(Path(vtt_path).resolve()).replace("\\", "/")
    escaped = (
        path_value
        .replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )
    return f"subtitles=filename='{escaped}'"


def strip_vtt_markup(text: str) -> str:
    text = html.unescape(text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " "))
    text = TAG_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def split_speaker_prefix(line: str) -> Optional[tuple[str, str]]:
    stripped = line.strip()
    for separator in SPEAKER_SEPARATORS:
        if separator not in stripped:
            continue

        speaker_part, text_part = stripped.split(separator, 1)
        speaker_candidate = normalize_speaker_name(speaker_part)
        if not speaker_candidate or speaker_candidate == "Unknown":
            continue
        if len(speaker_candidate) > 80:
            continue
        if len(speaker_candidate.split()) > 8:
            continue
        if any(char in speaker_candidate for char in ".!?"):
            continue
        if "-->" in speaker_candidate:
            continue

        return speaker_candidate, text_part.strip()

    return None


def parse_vtt_timestamp(value: str) -> float:
    raw = value.strip()
    parts = raw.split(":")
    if len(parts) == 3:
        hours_text, minutes_text, seconds_text = parts
    elif len(parts) == 2:
        hours_text = "0"
        minutes_text, seconds_text = parts
    else:
        raise VTTDubbingError(f"Timestamp không hợp lệ: '{value}'")

    if "." not in seconds_text:
        raise VTTDubbingError(f"Timestamp không hợp lệ: '{value}'")

    second_part, millisecond_part = seconds_text.split(".", 1)
    try:
        hours = int(hours_text)
        minutes = int(minutes_text)
        seconds = int(second_part)
        milliseconds = int(millisecond_part)
    except ValueError as exc:
        raise VTTDubbingError(f"Timestamp không hợp lệ: '{value}'") from exc

    if minutes < 0 or seconds < 0 or milliseconds < 0:
        raise VTTDubbingError(f"Timestamp không hợp lệ: '{value}'")
    if seconds >= 60 or milliseconds >= 1000:
        raise VTTDubbingError(f"Timestamp không hợp lệ: '{value}'")

    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0


def extract_speaker_and_text(text_lines: list[str]) -> tuple[str, str]:
    joined_raw = "\n".join(line.rstrip() for line in text_lines if line.strip())
    if not joined_raw.strip():
        return "Unknown", ""

    voice_match = VOICE_TAG_PATTERN.match(joined_raw)
    if voice_match:
        speaker = normalize_speaker_name(voice_match.group(1))
        text = strip_vtt_markup(voice_match.group(2))
        return speaker, text

    first_line = text_lines[0].strip()
    remainder_lines = [line.strip() for line in text_lines[1:] if line.strip()]

    bracket_match = BRACKET_SPEAKER_PATTERN.match(first_line)
    if bracket_match:
        speaker = normalize_speaker_name(bracket_match.group(1))
        text_parts = [bracket_match.group(2).strip(), *remainder_lines]
        return speaker, strip_vtt_markup(" ".join(part for part in text_parts if part))

    colon_match = split_speaker_prefix(first_line)
    if colon_match:
        speaker_candidate, first_text_part = colon_match
        text_parts = [first_text_part, *remainder_lines]
        return speaker_candidate, strip_vtt_markup(" ".join(part for part in text_parts if part))

    return "Unknown", strip_vtt_markup(" ".join(line.strip() for line in text_lines if line.strip()))


def parse_vtt_file(vtt_path: str | Path) -> list[VTTCue]:
    path = Path(vtt_path)
    if not path.exists():
        raise VTTDubbingError(f"Không tìm thấy file VTT: {path}")

    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise VTTDubbingError("File VTT không phải UTF-8 hợp lệ.") from exc

    if not content.strip():
        raise VTTDubbingError("File VTT rỗng.")

    lines = content.splitlines()
    cues: list[VTTCue] = []
    index = 0
    pointer = 0

    if lines and lines[0].strip().upper().startswith("WEBVTT"):
        pointer = 1

    while pointer < len(lines):
        current = lines[pointer].strip()
        if not current:
            pointer += 1
            continue

        if current.startswith("NOTE"):
            pointer += 1
            while pointer < len(lines) and lines[pointer].strip():
                pointer += 1
            continue

        if "-->" not in current:
            if pointer + 1 < len(lines) and "-->" in lines[pointer + 1]:
                pointer += 1
                current = lines[pointer].strip()
            else:
                pointer += 1
                continue

        if "-->" not in current:
            raise VTTDubbingError(f"Không tìm thấy mốc thời gian hợp lệ gần dòng {pointer + 1}.")

        start_raw, end_raw = current.split("-->", 1)
        start_seconds = parse_vtt_timestamp(start_raw)
        end_seconds = parse_vtt_timestamp(end_raw.strip().split()[0])
        if end_seconds <= start_seconds:
            raise VTTDubbingError(
                f"Mốc thời gian không hợp lệ tại cue {index + 1}: kết thúc phải lớn hơn bắt đầu."
            )

        pointer += 1
        cue_text_lines: list[str] = []
        while pointer < len(lines) and lines[pointer].strip():
            cue_text_lines.append(lines[pointer])
            pointer += 1

        speaker, text = extract_speaker_and_text(cue_text_lines)
        if text:
            index += 1
            cues.append(
                VTTCue(
                    index=index,
                    speaker=speaker,
                    text=text,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
            )

    if not cues:
        raise VTTDubbingError("Không phát hiện được cue hợp lệ trong file VTT.")

    return cues


def collect_speakers(cues: list[VTTCue]) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()
    for cue in cues:
        normalized = normalize_speaker_name(cue.speaker)
        if normalized not in seen:
            seen.add(normalized)
            speakers.append(normalized)
    return speakers


def export_detected_speakers(speakers: list[str], output_root: Path = OUTPUT_ROOT) -> Path:
    ensure_output_tree(output_root)
    export_path = output_root / "speakers_detected.json"
    payload = {"speakers": speakers}
    export_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return export_path


def validate_reference_audio_path(audio_path: str | None) -> Optional[Path]:
    if not audio_path:
        return None
    path = Path(audio_path)
    if not path.exists():
        raise VTTDubbingError(f"Reference audio không tồn tại: {path}")
    if path.suffix.lower() not in SUPPORTED_REFERENCE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_REFERENCE_SUFFIXES))
        raise VTTDubbingError(f"Reference audio không được hỗ trợ. Hãy dùng một trong các định dạng: {supported}")
    return path


def validate_video_path(video_path: str | None) -> Optional[Path]:
    if not video_path:
        return None
    path = Path(video_path)
    if not path.exists():
        raise VTTDubbingError(f"File MP4 không tồn tại: {path}")
    if path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise VTTDubbingError(f"Video input không được hỗ trợ. Hãy dùng một trong các định dạng: {supported}")
    return path


def detect_vtt_speakers(vtt_path: str | Path, output_root: Path = OUTPUT_ROOT) -> list[str]:
    cues = parse_vtt_file(vtt_path)
    speakers = collect_speakers(cues)
    export_detected_speakers(speakers, output_root=output_root)
    return speakers


def summarize_unknown_cues(cues: list[VTTCue], limit: int = 5) -> str:
    unknown_cues = [cue for cue in cues if normalize_speaker_name(cue.speaker) == "Unknown"]
    if not unknown_cues:
        return ""

    samples = []
    for cue in unknown_cues[:limit]:
        preview = cue.text[:90].strip()
        samples.append(f"cue {cue.index} @ {cue.start_seconds:.2f}s: {preview}")

    suffix = ""
    if len(unknown_cues) > limit:
        suffix = f" ... và thêm {len(unknown_cues) - limit} cue khác."

    return " | ".join(samples) + suffix


def ensure_ffmpeg(ffmpeg_path: str) -> None:
    try:
        completed = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise VTTDubbingError(
            "Không tìm thấy FFmpeg. Hãy cài FFmpeg và đảm bảo lệnh 'ffmpeg' hoạt động trong PATH, hoặc đặt biến môi trường FFMPEG_PATH."
        ) from exc

    if completed.returncode != 0:
        raise VTTDubbingError(f"FFmpeg không khởi động được: {completed.stderr.strip() or completed.stdout.strip()}")


def run_command(command: list[str], logger: RunLogger, error_context: str) -> None:
    logger.info(f"Chạy lệnh: {' '.join(command)}")
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "Không có stderr."
        logger.error(f"{error_context}: {stderr}")
        raise VTTDubbingError(f"{error_context}: {stderr}")


def run_command_capture_output(command: list[str], logger: RunLogger, error_context: str) -> str:
    logger.info(f"Chạy lệnh: {' '.join(command)}")
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "Không có stderr."
        logger.error(f"{error_context}: {stderr}")
        raise VTTDubbingError(f"{error_context}: {stderr}")
    return completed.stdout


def build_atempo_filter(speed: float) -> str:
    if speed <= 0:
        raise VTTDubbingError("Giá trị speed phải lớn hơn 0.")

    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5

    if abs(remaining - 1.0) > 1e-6 or not factors:
        factors.append(remaining)

    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def plan_cue_timing_adjustment(
    measured_duration: float,
    target_duration: float,
    max_speedup: float,
) -> CueTimingPlan:
    if measured_duration < 0:
        raise VTTDubbingError("Measured duration không được âm.")
    if target_duration <= 0:
        raise VTTDubbingError("Cue duration phải lớn hơn 0 để căn timing.")
    if max_speedup < 1.0:
        raise VTTDubbingError("Max speedup phải lớn hơn hoặc bằng 1.0.")

    required_ratio = measured_duration / target_duration
    min_ratio = 1.0 / max_speedup
    applied_ratio = min(max(required_ratio, min_ratio), max_speedup)
    return CueTimingPlan(
        target_duration=target_duration,
        measured_duration=measured_duration,
        required_ratio=required_ratio,
        applied_ratio=applied_ratio,
        min_ratio=min_ratio,
        max_ratio=max_speedup,
    )


def get_audio_duration(audio_path: Path) -> float:
    return float(sf.info(str(audio_path)).duration)


def apply_speed_with_ffmpeg(
    ffmpeg_path: str,
    input_path: Path,
    output_path: Path,
    speed: float,
    logger: RunLogger,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-af",
        build_atempo_filter(speed),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    run_command(command, logger, f"FFmpeg atempo thất bại cho {input_path.name}")


def truncate_audio_with_ffmpeg(
    ffmpeg_path: str,
    input_path: Path,
    output_path: Path,
    max_duration: float,
    logger: RunLogger,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-t",
        f"{max_duration:.3f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    run_command(command, logger, f"FFmpeg truncate thất bại cho {input_path.name}")


def safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def safe_rmtree(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def resolve_ffprobe_binary(ffmpeg_path: str) -> str:
    explicit = os.getenv("FFPROBE_PATH")
    if explicit:
        return explicit
    ffmpeg_candidate = Path(ffmpeg_path)
    if ffmpeg_candidate.parent != Path(".") or os.sep in ffmpeg_path:
        return str(ffmpeg_candidate.with_name("ffprobe"))
    return "ffprobe"


def probe_audio_stream_count(video_path: Path, ffprobe_path: str, logger: RunLogger) -> int:
    output = run_command_capture_output(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        logger,
        f"FFprobe không đọc được audio streams của {video_path.name}",
    )
    stream_lines = [line.strip() for line in output.splitlines() if line.strip()]
    return len(stream_lines)


def probe_subtitle_stream_count(video_path: Path, ffprobe_path: str, logger: RunLogger) -> int:
    output = run_command_capture_output(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        logger,
        f"FFprobe không đọc được subtitle streams của {video_path.name}",
    )
    stream_lines = [line.strip() for line in output.splitlines() if line.strip()]
    return len(stream_lines)


def to_serializable_voice_codes(codes: Any) -> Any:
    if "torch" in str(type(codes)):
        try:
            import torch

            if isinstance(codes, torch.Tensor):
                return codes.detach().cpu().numpy()
        except Exception:
            return codes
    return codes


def build_assignment_signature(
    speaker: str,
    voice_id: Optional[str],
    reference_audio: Optional[Path],
    reference_text: str,
    speed: float,
) -> str:
    payload = {
        "speaker": speaker,
        "voice_id": voice_id,
        "reference_audio": str(reference_audio) if reference_audio else None,
        "reference_audio_mtime_ns": reference_audio.stat().st_mtime_ns if reference_audio and reference_audio.exists() else None,
        "reference_audio_size": reference_audio.stat().st_size if reference_audio and reference_audio.exists() else None,
        "reference_text": reference_text,
        "speed": round(speed, 4),
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def prepare_speaker_assignments(
    tts_engine: Any,
    tts_lock: threading.Lock,
    is_turbo_model: bool,
    speakers: list[str],
    voice_ids: list[Any],
    reference_audio_paths: list[Any],
    reference_texts: list[Any],
    speeds: list[Any],
    logger: RunLogger,
) -> dict[str, SpeakerAssignment]:
    assignments: dict[str, SpeakerAssignment] = {}

    for idx, speaker in enumerate(speakers):
        raw_speaker = str(speaker).strip() if speaker is not None else ""
        if not raw_speaker:
            continue

        speaker_name = normalize_speaker_name(raw_speaker)
        voice_id = str(voice_ids[idx]).strip() if idx < len(voice_ids) and voice_ids[idx] else ""
        reference_audio = validate_reference_audio_path(str(reference_audio_paths[idx]) if idx < len(reference_audio_paths) and reference_audio_paths[idx] else None)
        reference_text = str(reference_texts[idx]).strip() if idx < len(reference_texts) and reference_texts[idx] else ""
        speed = float(speeds[idx]) if idx < len(speeds) else 1.0

        if speed < 0.7 or speed > 1.5:
            raise VTTDubbingError(f"Speed của speaker '{speaker_name}' phải nằm trong khoảng 0.7 đến 1.5.")

        if reference_audio is None and not voice_id:
            raise VTTDubbingError(
                f"Speaker '{speaker_name}' chưa được cấu hình giọng. Hãy chọn preset voice hoặc upload reference audio."
            )

        if reference_audio is not None and not is_turbo_model and not reference_text:
            raise VTTDubbingError(
                f"Speaker '{speaker_name}' đang dùng Voice Cloning ở Standard mode nên cần nhập Reference Text."
            )

        if reference_audio is not None:
            if voice_id:
                logger.warning(
                    f"Speaker '{speaker_name}' có cả preset voice và clone audio; hệ thống sẽ ưu tiên clone audio."
                )
            source_label = f"clone:{reference_audio.name}"
            logger.info(
                f"Speaker '{speaker_name}' dùng clone audio '{reference_audio.name}' với speed={speed:.2f}"
            )
            with tts_lock:
                voice_codes = to_serializable_voice_codes(tts_engine.encode_reference(reference_audio))
            voice_data = {"codes": voice_codes, "text": reference_text}
            assignments[speaker_name] = SpeakerAssignment(
                speaker=speaker_name,
                voice_data=voice_data,
                source_label=source_label,
                source_signature=build_assignment_signature(
                    speaker_name,
                    None,
                    reference_audio,
                    reference_text,
                    speed,
                ),
                speed=speed,
            )
            continue

        logger.info(f"Speaker '{speaker_name}' dùng preset voice '{voice_id}' với speed={speed:.2f}")
        with tts_lock:
            preset_voice = tts_engine.get_preset_voice(voice_id)
        preset_voice = {
            "codes": to_serializable_voice_codes(preset_voice["codes"]),
            "text": preset_voice["text"],
        }
        assignments[speaker_name] = SpeakerAssignment(
            speaker=speaker_name,
            voice_data=preset_voice,
            source_label=f"preset:{voice_id}",
            source_signature=build_assignment_signature(
                speaker_name,
                voice_id,
                None,
                preset_voice["text"],
                speed,
            ),
            speed=speed,
        )

    return assignments


def build_cache_payload(
    cue: VTTCue,
    assignment: SpeakerAssignment,
    selected_mode: str,
    emotion: str,
    overflow_mode: str,
    max_speedup: float,
) -> dict[str, Any]:
    return {
        "cache_schema_version": VTT_CACHE_SCHEMA_VERSION,
        "cue_index": cue.index,
        "speaker": cue.speaker,
        "text": cue.text,
        "start_seconds": round(cue.start_seconds, 3),
        "end_seconds": round(cue.end_seconds, 3),
        "mode": selected_mode,
        "emotion": emotion,
        "overflow_mode": overflow_mode,
        "max_speedup": round(max_speedup, 4),
        "source_label": assignment.source_label,
        "source_signature": assignment.source_signature,
        "speed": round(assignment.speed, 4),
        "cue_timing_strategy": "base_speed_plus_measured_auto_fit",
        "sample_rate": DEFAULT_SAMPLE_RATE,
    }


def read_cached_payload(meta_path: Path) -> Optional[dict[str, Any]]:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def finalize_audio_file(source_path: Path, target_path: Path) -> None:
    if source_path == target_path:
        return
    shutil.copyfile(source_path, target_path)


def render_single_cue(
    cue: VTTCue,
    assignment: SpeakerAssignment,
    tts_engine: Any,
    tts_lock: threading.Lock,
    ffmpeg_path: str,
    temp_audio_dir: Path,
    selected_mode: str,
    emotion: str,
    overflow_mode: str,
    max_speedup: float,
    logger: RunLogger,
) -> CueRenderResult:
    if not cue.text.strip():
        logger.info(f"Bỏ qua cue {cue.index} vì text rỗng.")
        return CueRenderResult(cue=cue, audio_path=None, duration_seconds=0.0, skipped=True)

    safe_speaker = sanitize_filename(cue.speaker)
    final_path = temp_audio_dir / f"cue_{cue.index:06d}__{safe_speaker}.wav"
    meta_path = temp_audio_dir / f"cue_{cue.index:06d}__{safe_speaker}.json"
    raw_path = temp_audio_dir / f"cue_{cue.index:06d}__{safe_speaker}__raw.wav"
    base_speed_path = temp_audio_dir / f"cue_{cue.index:06d}__{safe_speaker}__base.wav"
    auto_fit_path = temp_audio_dir / f"cue_{cue.index:06d}__{safe_speaker}__autofit.wav"
    overflow_path = temp_audio_dir / f"cue_{cue.index:06d}__{safe_speaker}__overflow.wav"

    cache_payload = build_cache_payload(cue, assignment, selected_mode, emotion, overflow_mode, max_speedup)
    cached_payload = read_cached_payload(meta_path)
    if final_path.exists() and cached_payload == cache_payload:
        duration = get_audio_duration(final_path)
        logger.info(f"Cache hit cho cue {cue.index}: {final_path.name}")
        return CueRenderResult(cue=cue, audio_path=final_path, duration_seconds=duration, skipped=False)

    for staged_path in (final_path, raw_path, base_speed_path, auto_fit_path, overflow_path):
        safe_unlink(staged_path)

    emotion_tag = "<|emotion_0|>" if emotion == "natural" and selected_mode == "standard" else None

    # Shared TTS instances inside VieNeu-TTS are not guaranteed to be thread-safe across backends.
    # We keep parallelism at the job level, but serialize infer()/encode_reference() through a lock.
    with tts_lock:
        audio = tts_engine.infer(
            cue.text,
            voice=assignment.voice_data,
            emotion_tag=emotion_tag,
            apply_watermark=True,
        )

    if audio is None or len(audio) == 0:
        raise VTTDubbingError(f"VieNeu không sinh được audio cho cue {cue.index}.")

    tts_engine.save(audio, raw_path)
    current_path = raw_path
    current_duration = get_audio_duration(current_path)

    if abs(assignment.speed - 1.0) > 1e-6:
        apply_speed_with_ffmpeg(ffmpeg_path, current_path, base_speed_path, assignment.speed, logger)
        current_path = base_speed_path
        current_duration = get_audio_duration(current_path)
        logger.info(
            f"Cue {cue.index}: áp dụng speaker speed nền {assignment.speed:.2f} cho '{cue.speaker}', duration={current_duration:.2f}s"
        )

    timing_plan = plan_cue_timing_adjustment(current_duration, cue.duration, max_speedup)
    if timing_plan.needs_adjustment:
        apply_speed_with_ffmpeg(ffmpeg_path, current_path, auto_fit_path, timing_plan.applied_ratio, logger)
        current_path = auto_fit_path
        current_duration = get_audio_duration(current_path)
        if timing_plan.was_clamped:
            logger.warning(
                f"Cue {cue.index}: auto-fit timing dùng ratio {timing_plan.applied_ratio:.2f}x "
                f"(yêu cầu {timing_plan.required_ratio:.2f}x, giới hạn tự nhiên {timing_plan.min_ratio:.2f}x-{timing_plan.max_ratio:.2f}x), "
                f"duration={current_duration:.2f}s / target={cue.duration:.2f}s"
            )
        else:
            logger.info(
                f"Cue {cue.index}: auto-fit timing với ratio {timing_plan.applied_ratio:.2f}x, "
                f"duration={current_duration:.2f}s / target={cue.duration:.2f}s"
            )

    if current_duration > cue.duration + CUE_TIMING_TOLERANCE_SECONDS:
        if overflow_mode == "speedup":
            speed_ratio = current_duration / cue.duration
            if speed_ratio > 1.0 + 1e-6:
                apply_speed_with_ffmpeg(ffmpeg_path, current_path, overflow_path, speed_ratio, logger)
                current_path = overflow_path
                current_duration = get_audio_duration(current_path)
                logger.warning(
                    f"Cue {cue.index}: vẫn vượt subtitle sau auto-fit tự nhiên, tăng tốc thêm {speed_ratio:.2f}x theo overflow_mode=speedup"
                )
        elif overflow_mode == "truncate":
            truncate_audio_with_ffmpeg(ffmpeg_path, current_path, overflow_path, cue.duration, logger)
            current_path = overflow_path
            current_duration = get_audio_duration(current_path)
            logger.warning(f"Cue {cue.index}: vẫn vượt subtitle sau auto-fit, đã truncate theo overflow_mode=truncate")
        else:
            logger.warning(f"Cue {cue.index}: vẫn vượt subtitle sau auto-fit nhưng giữ nguyên theo overflow_mode=keep")

    finalize_audio_file(current_path, final_path)
    meta_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for staged_path in (raw_path, base_speed_path, auto_fit_path, overflow_path):
        if staged_path != final_path:
            safe_unlink(staged_path)

    return CueRenderResult(cue=cue, audio_path=final_path, duration_seconds=current_duration, skipped=False)


def compose_with_ffmpeg(
    rendered_cues: list[CueRenderResult],
    output_path: Path,
    ffmpeg_path: str,
    logger: RunLogger,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    audio_cues = [item for item in rendered_cues if item.audio_path is not None]
    if not audio_cues:
        raise VTTDubbingError("Không có cue audio hợp lệ để ghép timeline.")

    total_duration = max(item.cue.end_seconds for item in rendered_cues)
    for item in audio_cues:
        total_duration = max(total_duration, item.cue.start_seconds + item.duration_seconds)
    total_duration += 0.05

    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-t",
        f"{total_duration:.3f}",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",
    ]

    filter_steps: list[str] = []
    mix_inputs = ["[0:a]"]
    for input_index, rendered in enumerate(audio_cues, start=1):
        command.extend(["-i", str(rendered.audio_path)])
        delay_ms = max(0, int(round(rendered.cue.start_seconds * 1000)))
        label = f"dub{input_index}"
        filter_steps.append(f"[{input_index}:a]adelay={delay_ms}|{delay_ms}[{label}]")
        mix_inputs.append(f"[{label}]")

    filter_steps.append(
        "".join(mix_inputs) + f"amix=inputs={len(mix_inputs)}:normalize=0:duration=longest[aout]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filter_steps),
            "-map",
            "[aout]",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(output_path),
        ]
    )

    run_command(command, logger, "FFmpeg ghép timeline thất bại")
    return output_path


def mux_dubbed_audio_into_video(
    ffmpeg_path: str,
    video_input_path: Path,
    dubbed_audio_path: Path,
    output_path: Path,
    logger: RunLogger,
    ffprobe_path: Optional[str] = None,
) -> Path:
    safe_unlink(output_path)
    ffprobe_binary = ffprobe_path or resolve_ffprobe_binary(ffmpeg_path)
    original_audio_count = probe_audio_stream_count(video_input_path, ffprobe_binary, logger)

    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_input_path),
        "-i",
        str(dubbed_audio_path),
        "-map_metadata",
        "0",
        "-map",
        "0:v?",
    ]

    if original_audio_count > 0:
        command.extend(["-map", "0:a:0"])

    command.extend(
        [
            "-map",
            "0:s?",
            "-map",
            "0:d?",
            "-map",
            "1:a:0",
            "-c",
            "copy",
        ]
    )

    dubbed_track_index = 1 if original_audio_count > 0 else 0
    command.extend(
        [
            f"-c:a:{dubbed_track_index}",
            "aac",
            f"-b:a:{dubbed_track_index}",
            "192k",
        ]
    )

    if original_audio_count > 0:
        command.extend(
            [
                "-disposition:a:0",
                "0",
                "-disposition:a:1",
                "default",
                "-metadata:s:a:0",
                "title=Original Audio",
                "-metadata:s:a:1",
                "title=Vietnamese Dubbed Audio",
                "-metadata:s:a:1",
                "language=vie",
            ]
        )
    else:
        command.extend(
            [
                "-disposition:a:0",
                "default",
                "-metadata:s:a:0",
                "title=Vietnamese Dubbed Audio",
                "-metadata:s:a:0",
                "language=vie",
            ]
        )

    command.append(str(output_path))
    run_command(command, logger, f"FFmpeg mux MP4 thất bại cho {video_input_path.name}")
    return output_path


def burn_vtt_subtitles_into_video(
    ffmpeg_path: str,
    input_video_path: Path,
    vtt_path: Path,
    output_path: Path,
    logger: RunLogger,
) -> Path:
    safe_unlink(output_path)
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video_path),
        "-vf",
        build_subtitles_filter(vtt_path),
        "-map_metadata",
        "0",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map",
        "0:d?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_command(command, logger, f"FFmpeg burn subtitle thất bại cho {input_video_path.name}")
    return output_path


def add_vtt_subtitle_track_to_video(
    ffmpeg_path: str,
    input_video_path: Path,
    vtt_path: Path,
    output_path: Path,
    logger: RunLogger,
    ffprobe_path: Optional[str] = None,
) -> Path:
    safe_unlink(output_path)
    ffprobe_binary = ffprobe_path or resolve_ffprobe_binary(ffmpeg_path)
    existing_subtitle_count = probe_subtitle_stream_count(input_video_path, ffprobe_binary, logger)
    subtitle_track_index = existing_subtitle_count

    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video_path),
        "-i",
        str(vtt_path),
        "-map_metadata",
        "0",
        "-map",
        "0:v?",
        "-map",
        "0:a?",
        "-map",
        "0:s?",
        "-map",
        "0:d?",
        "-map",
        "1:s:0",
        "-c",
        "copy",
        f"-c:s:{subtitle_track_index}",
        "mov_text",
        f"-metadata:s:s:{subtitle_track_index}",
        "title=Vietnamese Subtitle",
        f"-metadata:s:s:{subtitle_track_index}",
        "language=vie",
        str(output_path),
    ]
    run_command(command, logger, f"FFmpeg add subtitle track thất bại cho {input_video_path.name}")
    return output_path


def cleanup_vtt_artifacts(
    output_root: Path,
    logger: RunLogger,
    *,
    keep_paths: Optional[list[Path]] = None,
) -> None:
    keep_resolved = {path.resolve() for path in (keep_paths or []) if path.exists()}
    cleanup_candidates = [
        output_root / "temp",
        output_root / "logs",
        output_root / "speakers_detected.json",
    ]

    for candidate in cleanup_candidates:
        if candidate.resolve() in keep_resolved:
            continue
        if candidate.is_dir():
            safe_rmtree(candidate)
            continue
        safe_unlink(candidate)

    for candidate in (output_root / "temp", output_root / "logs"):
        try:
            if candidate.exists() and not any(candidate.iterdir()):
                candidate.rmdir()
        except OSError:
            pass


def generate_vtt_dubbing(
    *,
    tts_engine: Any,
    tts_lock: threading.Lock,
    vtt_path: str | Path,
    selected_mode: str,
    emotion: str,
    overflow_mode: str,
    max_speedup: float,
    worker_count: int,
    speakers: list[str],
    voice_ids: list[Any],
    reference_audio_paths: list[Any],
    reference_texts: list[Any],
    speeds: list[Any],
    logger: RunLogger,
    subtitle_export_mode: str = SUBTITLE_EXPORT_NONE,
    status_callback: Optional[Callable[[str], None]] = None,
    ffmpeg_path: Optional[str] = None,
    output_root: Path = OUTPUT_ROOT,
    output_path: Optional[str | Path] = None,
    video_input_path: Optional[str | Path] = None,
    video_output_path: Optional[str | Path] = None,
    cleanup_after_success: bool = True,
) -> Path:
    ensure_output_tree(output_root)
    ffmpeg_binary = ffmpeg_path or os.getenv("FFMPEG_PATH", "ffmpeg")
    ensure_ffmpeg(ffmpeg_binary)

    if selected_mode not in {"standard", "turbo"}:
        raise VTTDubbingError("Mode phải là 'standard' hoặc 'turbo'.")
    if emotion not in {"natural", "storytelling"}:
        raise VTTDubbingError("Emotion phải là 'natural' hoặc 'storytelling'.")
    if overflow_mode not in {"keep", "speedup", "truncate"}:
        raise VTTDubbingError("Overflow mode phải là keep, speedup hoặc truncate.")
    if worker_count < 1 or worker_count > 8:
        raise VTTDubbingError("Số worker threads phải nằm trong khoảng 1 đến 8.")
    if max_speedup < 1.0 or max_speedup > 3.0:
        raise VTTDubbingError("Max speedup phải nằm trong khoảng 1.0 đến 3.0.")

    subtitle_mode = validate_subtitle_export_mode(subtitle_export_mode)
    validated_video_input = validate_video_path(str(video_input_path) if video_input_path else None)
    if subtitle_mode != SUBTITLE_EXPORT_NONE and validated_video_input is None:
        raise VTTDubbingError("Muốn export subtitle vào video thì cần upload thêm file MP4.")
    planned_video_output = Path(video_output_path) if video_output_path else (
        build_vtt_video_output_path(vtt_path, output_root=output_root) if validated_video_input else None
    )

    cues = parse_vtt_file(vtt_path)
    speakers_detected = collect_speakers(cues)
    export_detected_speakers(speakers_detected, output_root=output_root)
    unknown_summary = summarize_unknown_cues(cues)

    logger.info(f"VTT path: {Path(vtt_path).resolve()}")
    logger.info(f"Số speaker phát hiện: {len(speakers_detected)}")
    logger.info(f"Số cue hợp lệ: {len(cues)}")
    logger.info(
        f"Mode={selected_mode}, emotion={emotion}, overflow_mode={overflow_mode}, "
        f"max_speedup={max_speedup:.2f}, subtitle_export_mode={subtitle_mode}"
    )
    logger.info(f"Worker threads={worker_count}")
    if unknown_summary:
        logger.warning(
            "Phát hiện cue không match speaker format và bị gán 'Unknown'. "
            f"Chi tiết: {unknown_summary}"
        )

    actual_speaker_inputs = [normalize_speaker_name(name) for name in speakers if str(name).strip()]
    missing_speakers = [speaker for speaker in speakers_detected if speaker not in actual_speaker_inputs]
    if missing_speakers:
        extra_detail = ""
        if "Unknown" in missing_speakers and unknown_summary:
            extra_detail = f" Các cue bị gán Unknown: {unknown_summary}"
        raise VTTDubbingError(
            f"Thiếu cấu hình cho speaker: {', '.join(missing_speakers)}. Hãy Detect Speakers lại và map đủ giọng.{extra_detail}"
        )

    assignments = prepare_speaker_assignments(
        tts_engine=tts_engine,
        tts_lock=tts_lock,
        is_turbo_model=(selected_mode == "turbo"),
        speakers=speakers,
        voice_ids=voice_ids,
        reference_audio_paths=reference_audio_paths,
        reference_texts=reference_texts,
        speeds=speeds,
        logger=logger,
    )

    temp_audio_dir = output_root / "temp" / "audio"
    rendered_results: dict[int, CueRenderResult] = {}

    def emit_status() -> None:
        if status_callback is not None:
            status_callback(logger.render())

    emit_status()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_cue = {
            executor.submit(
                render_single_cue,
                cue,
                assignments[normalize_speaker_name(cue.speaker)],
                tts_engine,
                tts_lock,
                ffmpeg_binary,
                temp_audio_dir,
                selected_mode,
                emotion,
                overflow_mode,
                max_speedup,
                logger,
            ): cue
            for cue in cues
            if cue.text.strip()
        }

        completed_count = 0
        total_count = len(future_to_cue)
        for future in as_completed(future_to_cue):
            cue = future_to_cue[future]
            result = future.result()
            rendered_results[cue.index] = result
            completed_count += 1
            logger.info(
                f"Tiến trình: {completed_count}/{total_count} cue hoàn tất ({cue.speaker} @ {cue.start_seconds:.2f}s)"
            )
            emit_status()

    ordered_results = [rendered_results[index] for index in sorted(rendered_results)]
    final_output_path = Path(output_path) if output_path else build_vtt_output_path(vtt_path, output_root=output_root)
    compose_with_ffmpeg(ordered_results, final_output_path, ffmpeg_binary, logger)
    logger.info(f"Hoàn tất. Final audio: {final_output_path.resolve()}")

    if validated_video_input is not None and planned_video_output is not None:
        temp_video_root = output_root / "temp"
        muxed_base_video_path = temp_video_root / f"{final_output_path.stem}__base_mux.mp4"
        mux_dubbed_audio_into_video(
            ffmpeg_path=ffmpeg_binary,
            video_input_path=validated_video_input,
            dubbed_audio_path=final_output_path,
            output_path=muxed_base_video_path if subtitle_mode != SUBTITLE_EXPORT_NONE else planned_video_output,
            logger=logger,
        )
        if subtitle_mode == SUBTITLE_EXPORT_BURN:
            burn_vtt_subtitles_into_video(
                ffmpeg_path=ffmpeg_binary,
                input_video_path=muxed_base_video_path,
                vtt_path=Path(vtt_path),
                output_path=planned_video_output,
                logger=logger,
            )
        elif subtitle_mode == SUBTITLE_EXPORT_SOFT:
            add_vtt_subtitle_track_to_video(
                ffmpeg_path=ffmpeg_binary,
                input_video_path=muxed_base_video_path,
                vtt_path=Path(vtt_path),
                output_path=planned_video_output,
                logger=logger,
            )
        logger.info(f"Hoàn tất MP4 mux: {planned_video_output.resolve()}")

    if cleanup_after_success:
        logger.info("Final outputs đã sẵn sàng. Bắt đầu dọn dẹp temp/cache để tiết kiệm dung lượng.")
        emit_status()
        cleanup_keep_paths = [final_output_path]
        if planned_video_output is not None and planned_video_output.exists():
            cleanup_keep_paths.append(planned_video_output)
        cleanup_vtt_artifacts(output_root, logger, keep_paths=cleanup_keep_paths)

    emit_status()
    return final_output_path
