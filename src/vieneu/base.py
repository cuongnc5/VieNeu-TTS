from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import json
import math
import re
import numpy as np
import logging
from huggingface_hub import hf_hub_download
from sea_g2p import Normalizer
from vieneu_utils.core_utils import env_bool
from .logging_utils import configure_dubbing_diagnostic_logger
from .text_safety import ShortCueRewritePlan, rewrite_problematic_short_cue

# Configure logging
logger = logging.getLogger("Vieneu")

class BaseVieneuTTS(ABC):
    """
    Abstract base class for VieNeu-TTS implementations.
    Provides shared functionality for voice management and common operations.
    """

    def __init__(self, codec_repo: Optional[str] = None, codec_device: str = "cpu"):
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480
        self._diagnostic_console_logging = env_bool("VIENEU_DIAGNOSTIC_LOGGING", default=False)
        self._diagnostic_file_logging = env_bool("VIENEU_FILE_DIAGNOSTIC_LOGGING", default=True)
        self._diagnostic_file_logger = (
            configure_dubbing_diagnostic_logger()
            if self._diagnostic_file_logging
            else None
        )

        # Default streaming parameters
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 50
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        self.assets_dir = Path(__file__).parent / "assets"
        self._preset_voices: Dict[str, Any] = {}
        self._default_voice: Optional[str] = None
        self.normalizer = Normalizer()
        self._ref_phoneme_cache: Dict[str, str] = {}

        # Watermarker placeholder
        self.watermarker = None
        self._init_watermarker()

        if codec_repo:
            self._load_codec(codec_repo, codec_device)

    def _should_log_diagnostics(self, log: Optional[logging.Logger] = None) -> bool:
        active_logger = log or logger
        return self._diagnostic_console_logging or active_logger.isEnabledFor(logging.DEBUG)

    def _diagnostics_enabled(self, log: Optional[logging.Logger] = None) -> bool:
        return self._diagnostic_file_logger is not None or self._should_log_diagnostics(log)

    def _emit_diagnostic_file(self, log: logging.Logger, level: int, message: str, *args: Any) -> None:
        if self._diagnostic_file_logger is None:
            return
        self._diagnostic_file_logger.log(level, f"[{log.name}] {message}", *args)

    def _emit_diagnostic(self, log: logging.Logger, message: str, *args: Any) -> None:
        self._emit_diagnostic_file(log, logging.INFO, message, *args)
        if not self._should_log_diagnostics(log):
            return
        if self._diagnostic_console_logging:
            log.info(message, *args)
        else:
            log.debug(message, *args)

    def _emit_diagnostic_warning(self, log: logging.Logger, message: str, *args: Any) -> None:
        self._emit_diagnostic_file(log, logging.WARNING, message, *args)
        if self._diagnostic_console_logging or log.isEnabledFor(logging.DEBUG):
            log.warning(message, *args)

    def _shorten_for_log(self, value: Optional[str], limit: int = 180) -> str:
        if value is None:
            return "<none>"
        clean = " ".join(str(value).split())
        return clean if len(clean) <= limit else f"{clean[:limit - 3]}..."

    def _count_words(self, text: Optional[str]) -> int:
        if not text:
            return 0
        return len([part for part in text.split() if part])

    def _longest_repeat_run(self, values: List[int]) -> int:
        if not values:
            return 0
        longest = 1
        current = 1
        for prev, curr in zip(values, values[1:]):
            if prev == curr:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return longest

    def _preview_token_ids(self, token_ids: List[int], limit: int = 12) -> str:
        if not token_ids:
            return "[]"
        preview = ", ".join(str(token) for token in token_ids[:limit])
        suffix = ", ..." if len(token_ids) > limit else ""
        return f"[{preview}{suffix}]"

    def _speech_ids_to_string(self, speech_ids: List[int]) -> str:
        return "".join(f"<|speech_{token}|>" for token in speech_ids)

    def _speech_token_seconds(self) -> float:
        return self.hop_length / self.sample_rate

    def _normalize_guard_tokens(self, value: str) -> List[str]:
        tokens = []
        for part in value.split():
            cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", part, flags=re.UNICODE).strip().lower()
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def _find_repeat_run(self, speech_ids: List[int], threshold: int) -> tuple[Optional[int], int]:
        if not speech_ids:
            return None, 0
        run_start = 0
        run_length = 1
        for index in range(1, len(speech_ids)):
            if speech_ids[index] == speech_ids[index - 1]:
                run_length += 1
                if run_length >= threshold:
                    return run_start, run_length
            else:
                run_start = index
                run_length = 1
        return None, 0

    def _build_generation_guard(
        self,
        chunk_text: str,
        chunk_phonemes: str,
        *,
        target_duration_s: Optional[float] = None,
        requested_max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        word_count = max(self._count_words(chunk_text), 1)
        phoneme_word_count = max(self._count_words(chunk_phonemes), 1)
        effective_word_count = max(word_count, phoneme_word_count)
        text_tokens = self._normalize_guard_tokens(chunk_text)
        phoneme_tokens = self._normalize_guard_tokens(chunk_phonemes)
        effective_tokens = phoneme_tokens or text_tokens
        unique_token_count = len(set(effective_tokens)) if effective_tokens else effective_word_count
        repetitive_compact = effective_word_count <= 4 and unique_token_count <= 2
        short_text = effective_word_count <= 3 or repetitive_compact
        repeat_threshold = 12 if short_text else 16

        desired_max_duration_s = 0.8 + 0.4 * effective_word_count
        desired_max_duration_s = max(1.2, min(desired_max_duration_s, 9.0))

        if repetitive_compact:
            desired_max_duration_s = min(desired_max_duration_s, 2.4)
        elif short_text:
            desired_max_duration_s = min(desired_max_duration_s, 2.8)

        if target_duration_s is not None and target_duration_s > 0:
            target_ceiling = min(target_duration_s + 0.4, target_duration_s * 1.25)
            desired_max_duration_s = min(desired_max_duration_s, max(1.0, target_ceiling))

        token_seconds = self._speech_token_seconds()
        hard_cap = requested_max_new_tokens if requested_max_new_tokens is not None else 2048
        speech_token_cap = int(math.ceil(desired_max_duration_s / token_seconds)) + 10
        max_new_tokens = min(speech_token_cap, hard_cap)
        min_new_tokens = min(max(12, int(math.ceil(min(desired_max_duration_s, 1.4) / token_seconds))), max(max_new_tokens - 8, 1))

        return {
            "word_count": word_count,
            "phoneme_word_count": phoneme_word_count,
            "short_text": short_text,
            "repetitive_compact": repetitive_compact,
            "unique_token_count": unique_token_count,
            "max_new_tokens": int(max_new_tokens),
            "min_new_tokens": int(min_new_tokens),
            "speech_token_cap": int(max_new_tokens),
            "repeat_threshold": repeat_threshold,
            "target_duration_s": target_duration_s,
            "desired_max_duration_s": round(desired_max_duration_s, 3),
        }

    def _apply_generation_guard(
        self,
        log: logging.Logger,
        *,
        chunk_text: str,
        chunk_phonemes: str,
        output_str: str,
        guard: Dict[str, Any],
    ) -> str:
        from .utils import extract_speech_ids

        speech_ids = extract_speech_ids(output_str)
        if not speech_ids:
            return output_str

        speech_token_cap = int(guard.get("speech_token_cap", 2048))
        repeat_threshold = int(guard.get("repeat_threshold", 16))
        repeat_start, repeat_length = self._find_repeat_run(speech_ids, repeat_threshold)
        cut_index: Optional[int] = None
        reasons: List[str] = []

        if len(speech_ids) > speech_token_cap:
            cut_index = speech_token_cap
            reasons.append(f"cap={speech_token_cap}")

        minimum_repeat_prefix = max(24, min(speech_token_cap // 2, 80))
        if repeat_start is not None and repeat_start >= minimum_repeat_prefix:
            if cut_index is None or repeat_start < cut_index:
                cut_index = repeat_start
            reasons.append(f"repeat_run={repeat_length}")

        if cut_index is None or cut_index >= len(speech_ids):
            return output_str

        trimmed_ids = speech_ids[:cut_index]
        self._emit_diagnostic_warning(
            log,
            "[guard] trimmed speech output reasons=%s kept_tokens=%d dropped_tokens=%d text=%s phonemes=%s token_preview=%s",
            ",".join(reasons),
            len(trimmed_ids),
            len(speech_ids) - len(trimmed_ids),
            self._shorten_for_log(chunk_text, limit=120),
            self._shorten_for_log(chunk_phonemes, limit=180),
            self._preview_token_ids(trimmed_ids, limit=24),
        )
        return self._speech_ids_to_string(trimmed_ids)

    def _build_sampling_controls(
        self,
        *,
        temperature: float,
        top_k: int,
        guard: Dict[str, Any],
        repetition_penalty: Optional[float] = None,
    ) -> Dict[str, Any]:
        constrained = bool(guard.get("short_text"))
        highly_constrained = bool(guard.get("repetitive_compact"))

        if highly_constrained:
            adjusted_temperature = min(temperature, 0.15)
            adjusted_top_k = max(1, min(top_k, 8))
            adjusted_repetition_penalty = (
                max(repetition_penalty, 1.25) if repetition_penalty is not None else 1.25
            )
        elif constrained:
            adjusted_temperature = min(temperature, 0.25)
            adjusted_top_k = max(1, min(top_k, 16))
            adjusted_repetition_penalty = (
                max(repetition_penalty, 1.18) if repetition_penalty is not None else 1.18
            )
        else:
            adjusted_temperature = temperature
            adjusted_top_k = top_k
            adjusted_repetition_penalty = repetition_penalty

        return {
            "temperature": adjusted_temperature,
            "top_k": adjusted_top_k,
            "repetition_penalty": adjusted_repetition_penalty,
            "constrained": constrained,
            "highly_constrained": highly_constrained,
        }

    def _prepare_text_for_inference(
        self,
        log: logging.Logger,
        text: str,
        *,
        rewrite_enabled: bool = True,
    ) -> tuple[str, ShortCueRewritePlan]:
        default_plan = ShortCueRewritePlan(
            original_text=text,
            normalized_text="",
            rewritten_text=text,
            applied=False,
        )
        if not rewrite_enabled:
            return text, default_plan

        rewrite_plan = rewrite_problematic_short_cue(text)
        if not rewrite_plan.applied:
            return text, rewrite_plan

        self._emit_diagnostic_file(
            log,
            logging.INFO,
            "[guard] rewrite problematic short text original=%s rewritten=%s reason=%s",
            self._shorten_for_log(rewrite_plan.original_text, limit=120),
            self._shorten_for_log(rewrite_plan.rewritten_text, limit=120),
            rewrite_plan.reason or "unknown",
        )
        log.info(
            "Rewrite problematic short text for stable synthesis: %r -> %r (reason=%s)",
            rewrite_plan.original_text,
            rewrite_plan.rewritten_text,
            rewrite_plan.reason or "unknown",
        )
        return rewrite_plan.rewritten_text, rewrite_plan

    def _prepare_texts_for_inference(
        self,
        log: logging.Logger,
        texts: List[str],
        *,
        rewrite_enabled: bool = True,
    ) -> List[str]:
        return [
            self._prepare_text_for_inference(log, text, rewrite_enabled=rewrite_enabled)[0]
            for text in texts
        ]

    def _log_reference_context(
        self,
        log: logging.Logger,
        ref_text: Optional[str],
        ref_phonemes: Optional[str],
        ref_codes: Optional[Any],
    ) -> None:
        if not self._diagnostics_enabled(log):
            return
        ref_code_count = 0
        if ref_codes is not None:
            try:
                ref_code_count = len(self.to_list(ref_codes))
            except Exception:
                ref_code_count = 0
        self._emit_diagnostic(
            log,
            "[diag] reference ref_text=%s ref_phonemes=%s ref_codes=%d",
            self._shorten_for_log(ref_text, limit=120),
            self._shorten_for_log(ref_phonemes, limit=160),
            ref_code_count,
        )

    def _log_text_preparation(
        self,
        log: logging.Logger,
        original_text: str,
        normalized_text: str,
        chunks: List[str],
    ) -> None:
        if not self._diagnostics_enabled(log):
            return
        chunk_preview = " | ".join(self._shorten_for_log(chunk, limit=72) for chunk in chunks[:3]) or "<empty>"
        if len(chunks) > 3:
            chunk_preview = f"{chunk_preview} | ...(+{len(chunks) - 3} more)"
        self._emit_diagnostic(
            log,
            "[diag] input text=%s normalized=%s chunks=%d chunk_preview=%s",
            self._shorten_for_log(original_text),
            self._shorten_for_log(normalized_text),
            len(chunks),
            chunk_preview,
        )

    def _log_phoneme_preparation(
        self,
        log: logging.Logger,
        original_text: str,
        phonemes: str,
        chunks: List[str],
    ) -> None:
        if not self._diagnostics_enabled(log):
            return
        chunk_preview = " | ".join(self._shorten_for_log(chunk, limit=72) for chunk in chunks[:3]) or "<empty>"
        if len(chunks) > 3:
            chunk_preview = f"{chunk_preview} | ...(+{len(chunks) - 3} more)"
        self._emit_diagnostic(
            log,
            "[diag] input text=%s phonemes=%s chunks=%d chunk_preview=%s",
            self._shorten_for_log(original_text),
            self._shorten_for_log(phonemes),
            len(chunks),
            chunk_preview,
        )

    def _log_chunk_inputs(
        self,
        log: logging.Logger,
        *,
        chunk_index: int,
        total_chunks: int,
        chunk_text: str,
        chunk_phonemes: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._diagnostics_enabled(log):
            return
        extra_str = ""
        if extra:
            rendered = " ".join(f"{key}={value}" for key, value in extra.items())
            extra_str = f" {rendered}"
        self._emit_diagnostic(
            log,
            "[diag] chunk=%d/%d text=%s phonemes=%s words=%d phoneme_words=%d%s",
            chunk_index + 1,
            total_chunks,
            self._shorten_for_log(chunk_text, limit=120),
            self._shorten_for_log(chunk_phonemes, limit=180),
            self._count_words(chunk_text),
            self._count_words(chunk_phonemes),
            extra_str,
        )

    def _log_generation_result(
        self,
        log: logging.Logger,
        *,
        chunk_index: int,
        total_chunks: int,
        chunk_text: str,
        chunk_phonemes: str,
        output_str: str,
        wav: Optional[np.ndarray] = None,
    ) -> None:
        if not self._diagnostics_enabled(log):
            return

        from .utils import extract_speech_ids

        speech_ids = extract_speech_ids(output_str)
        speech_token_count = len(speech_ids)
        estimated_duration_s = speech_token_count * self._speech_token_seconds()
        actual_duration_s = (len(wav) / self.sample_rate) if wav is not None and len(wav) > 0 else 0.0
        word_count = max(self._count_words(chunk_text), 1)
        seconds_per_word = estimated_duration_s / word_count
        longest_repeat_run = self._longest_repeat_run(speech_ids)
        unique_ratio = len(set(speech_ids)) / speech_token_count if speech_token_count else 0.0

        self._emit_diagnostic(
            log,
            "[diag] chunk=%d/%d speech_tokens=%d est_sec=%.2f audio_sec=%.2f sec_per_word=%.2f longest_repeat_run=%d unique_ratio=%.3f token_preview=%s",
            chunk_index + 1,
            total_chunks,
            speech_token_count,
            estimated_duration_s,
            actual_duration_s,
            seconds_per_word,
            longest_repeat_run,
            unique_ratio,
            self._preview_token_ids(speech_ids),
        )

        reasons: List[str] = []
        if speech_token_count == 0:
            reasons.append("no_speech_tokens")
        if seconds_per_word > 1.35:
            reasons.append(f"sec_per_word={seconds_per_word:.2f}")
        if longest_repeat_run >= 8:
            reasons.append(f"repeat_run={longest_repeat_run}")
        if speech_token_count >= max(160, word_count * 70):
            reasons.append(f"speech_tokens={speech_token_count}")
        if speech_token_count >= 40 and unique_ratio <= 0.20:
            reasons.append(f"low_unique_ratio={unique_ratio:.3f}")

        if reasons:
            self._emit_diagnostic_warning(
                log,
                "[diag][warn] chunk=%d/%d possible_elongation reasons=%s text=%s phonemes=%s token_preview=%s",
                chunk_index + 1,
                total_chunks,
                ",".join(reasons),
                self._shorten_for_log(chunk_text, limit=120),
                self._shorten_for_log(chunk_phonemes, limit=180),
                self._preview_token_ids(speech_ids, limit=24),
            )

    def _load_codec(self, codec_repo: str, codec_device: str) -> None:
        """Universal codec loader for all backends."""
        logger.info(f"📦 Loading codec from: {codec_repo} on {codec_device} ...")

        if any(x in codec_repo.lower() for x in ["onnx", "vieneu-codec"]) or codec_repo == "neuphonic/neucodec-onnx-decoder-int8":
            if codec_device != "cpu":
                logger.warning("⚠️ ONNX decoder only runs on CPU. Ignoring device selection.")
            try:
                from .utils import NeuCodecOnnx
                self.codec = NeuCodecOnnx.from_pretrained(codec_repo)
                self._is_onnx_codec = True
                return
            except Exception as e:
                logger.warning(f"Failed to load standalone ONNX decoder: {e}. Trying via neucodec package...")
                try:
                    from neucodec import NeuCodecOnnxDecoder
                    self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                    self._is_onnx_codec = True
                    return
                except ImportError:
                    raise ImportError(
                        "The 'onnxruntime' package is required for ONNX decoder. \n"
                        "Please install it via: pip install onnxruntime"
                    ) from e

        # For PyTorch codecs, check for torch first
        try:
            import torch
            from neucodec import NeuCodec, DistillNeuCodec
            
            # Check MPS
            if codec_device == "mps" and not torch.backends.mps.is_available():
                logger.warning("⚠️ MPS not available for codec, falling back to CPU")
                codec_device = "cpu"

            if codec_repo == "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
            elif codec_repo == "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
            else:
                raise ValueError(f"Unrecognized codec repository: {codec_repo}")

            self.codec.eval().to(codec_device)
        except ImportError:
            raise ImportError(
                f"Codec '{codec_repo}' requires PyTorch. \n"
                "To remain lightweight in Remote mode, please use 'neuphonic/neucodec-onnx-decoder-int8'. \n"
                "Or install torch via: pip install vieneu[gpu]"
            )


    def _init_watermarker(self) -> None:
        """Initialize optional audio watermarker."""
        try:
            import perth
            self.watermarker = perth.PerthImplicitWatermarker()
            logger.info("🔒 Audio watermarking initialized (Perth)")
        except (ImportError, AttributeError):
            self.watermarker = None

    def _load_voices(self, backbone_repo: Optional[str], hf_token: Optional[str] = None, clear_existing: bool = False) -> None:
        """Unified voice loading for Local and Remote paths."""
        if not backbone_repo:
            return

        path_obj = Path(backbone_repo)
        if path_obj.exists():
            # Local Path (Dir or File)
            if path_obj.is_dir():
                json_path = path_obj / "voices.json"
            else:
                json_path = path_obj.parent / "voices.json"

            if json_path.exists():
                self._load_voices_from_file(json_path, clear_existing=clear_existing)
            else:
                if clear_existing:
                     self._preset_voices.clear()
                logger.warning(f"Validation Warning: Local path '{backbone_repo}' missing 'voices.json'.")
                logger.warning(f"Falling back to Custom Voice Cloning mode.")
        else:
            # Remote Repo
            if clear_existing:
                self._preset_voices.clear()

            try:
                self._load_voices_from_repo(backbone_repo, hf_token)
            except Exception as e:
                logger.warning(f"Could not load voices from repo '{backbone_repo}': {e}")
                logger.warning(f"Falling back to Custom Voice Cloning mode.")

    def _load_voices_from_file(self, file_path: Path, clear_existing: bool = False) -> None:
        """Load voices from a local JSON file."""
        try:
            if not file_path.exists():
                logger.error(f"Voice file not found: {file_path}")
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in voice file {file_path}: {e}")
                    return

            if "presets" in data:
                if clear_existing:
                    self._preset_voices.clear()
                    logger.info("🧹 Cleared existing voices for replacement")

                # Merge into existing presets
                self._preset_voices.update(data["presets"])
                logger.info(f"📢 Loaded {len(data['presets'])} voices from {file_path.name}")

            # Update default voice if provided
            if "default_voice" in data and data["default_voice"]:
                self._default_voice = data["default_voice"]

        except Exception as e:
            logger.error(f"Failed to load voices from {file_path}: {e}")

    def _load_voices_from_repo(self, repo_id: str, hf_token: Optional[str] = None) -> None:
        """Download and load voices.json from a HuggingFace repo."""
        voices_file = None
        try:
            # 1. Try normal download (checks for updates from server)
            voices_file = hf_hub_download(
                repo_id=repo_id,
                filename="voices.json",
                token=hf_token,
                repo_type="model"
            )
        except Exception:
            # 2. Network error? Try to use cached version if available
            logger.warning(f"Network check failed for voices.json. Trying local cache...")
            try:
                voices_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="voices.json",
                    token=hf_token,
                    repo_type="model",
                    local_files_only=True
                )
                logger.info(f"✅ Using cached voices.json")
            except Exception:
                # 3. No cache available either
                pass

        if voices_file:
            self._load_voices_from_file(Path(voices_file))
        else:
            logger.warning(f"Repository '{repo_id}' is missing 'voices.json'. Falling back to Custom Voice mode.")

    def list_preset_voices(self) -> List[tuple[str, str]]:
        """List available preset voices as (description, id)."""
        return [
            (v.get("description", k) if isinstance(v, dict) else str(v), k)
            for k, v in self._preset_voices.items()
        ]

    def get_preset_voice(self, voice_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get reference codes and text for a preset voice.

        Args:
            voice_name: Name of voice. If None, uses default_voice.

        Returns:
            dict: { 'codes': Union[np.ndarray, 'torch.Tensor'], 'text': str }
        """
        if voice_name is None:
            voice_name = self._default_voice
            if voice_name is None:
                if self._preset_voices:
                    voice_name = next(iter(self._preset_voices))
                else:
                    raise ValueError("No voice specified and no preset voices available.")

        if voice_name not in self._preset_voices:
            raise ValueError(f"Voice '{voice_name}' not found. Available: {self.list_preset_voices()}")

        voice_data = self._preset_voices[voice_name]
        codes = voice_data["codes"]
        
        # Only convert to torch if explicitly requested or if we're not in turbo mode
        if isinstance(codes, list):
            if codes and isinstance(codes[0], float):
                codes = np.array(codes, dtype=np.float32)
            else:
                # Là integer token sequence (Standard mode)
                try:
                    import torch
                    codes = torch.tensor(codes, dtype=torch.long)
                except ImportError:
                    codes = np.array(codes, dtype=np.int64)

        return {"codes": codes, "text": voice_data["text"]}

    def get_ref_phonemes(self, ref_text: str) -> str:
        """
        Get phonemized version of reference text, using cache if available.
        """
        if ref_text not in self._ref_phoneme_cache:
            from vieneu_utils.phonemize_text import phonemize_with_dict
            self._ref_phoneme_cache[ref_text] = phonemize_with_dict(ref_text)
        return self._ref_phoneme_cache[ref_text]

    def save(self, audio: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save audio waveform to a file."""
        import soundfile as sf
        sf.write(str(output_path), audio, self.sample_rate)

    def encode_reference(self, ref_audio_path: Union[str, Path]) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Encode reference audio to codes.

        Args:
            ref_audio_path: Path to the reference audio file.

        Returns:
            Union[np.ndarray, torch.Tensor]: Encoded codes.
        """
        import librosa
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        
        # If we have an ONNX encoder or specialized turbo encoder, handle it here
        # For now, default backends still use torch
        try:
            import torch
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]

            # Ensure device and dtype compatibility
            if hasattr(self.codec, "device"):
                wav_tensor = wav_tensor.to(self.codec.device)

            with torch.no_grad():
                ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
            return ref_codes
        except ImportError:
            raise ImportError("Torch is required for encode_reference in the current backend. Please install torch or use a backend that supports standalone encoding.")

    def _decode(self, codes_str: str) -> np.ndarray:
        """
        Decode speech tokens to audio waveform.

        Args:
            codes_str: String containing speech tokens.

        Returns:
            np.ndarray: Decoded audio waveform.
        """
        from .utils import extract_speech_ids
        speech_ids = extract_speech_ids(codes_str)

        if len(speech_ids) == 0:
            raise ValueError("No valid speech tokens found in the output.")

        # Onnx decode
        if getattr(self, "_is_onnx_codec", False):
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes)
        # Torch decode
        else:
            try:
                import torch
                with torch.no_grad():
                    codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :]
                    if hasattr(self.codec, "device"):
                        codes = codes.to(self.codec.device)

                    recon = self.codec.decode_code(codes)
                    if hasattr(recon, "cpu"):
                        recon = recon.cpu()
                    if hasattr(recon, "numpy"):
                        recon = recon.numpy()
            except ImportError:
                raise ImportError("Torch is required for the current codec backend. Please install torch or use an ONNX-based codec.")


        return recon[0, 0, :]

    def _resolve_ref_voice(
        self,
        voice: Optional[Dict[str, Any]] = None,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_codes: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
        ref_text: Optional[str] = None
    ) -> tuple[Union[np.ndarray, 'torch.Tensor'], str]:
        """Resolve reference voice codes and text."""
        if voice is not None:
            ref_codes = voice.get('codes', ref_codes)
            ref_text = voice.get('text', ref_text)

        if ref_audio is not None and ref_codes is None:
            ref_codes = self.encode_reference(ref_audio)
        elif self._default_voice and (ref_codes is None or ref_text is None):
            try:
                voice_data = self.get_preset_voice(None)
                ref_codes = voice_data['codes']
                ref_text = voice_data['text']
            except Exception:
                pass

        if ref_codes is None or ref_text is None:
            raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")

        return ref_codes, ref_text

    def _apply_watermark(self, wav: np.ndarray) -> np.ndarray:
        """Apply watermark to audio if enabled."""
        if self.watermarker:
            return self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
        return wav

    def to_list(self, codes: Any) -> List[int]:
        """Convert reference codes (Tensor, Array, List) to a Python list of integers."""
        if isinstance(codes, list):
            return codes
        if isinstance(codes, np.ndarray):
            return codes.flatten().tolist()

        # Check for torch without importing it at module level
        try:
            import torch
            if isinstance(codes, torch.Tensor):
                return codes.flatten().tolist()
        except ImportError:
            pass

        # Fallback for other array-like types
        if hasattr(codes, "tolist"):
            return codes.flatten().tolist() if hasattr(codes, "flatten") else codes.tolist()

        return list(codes)

    def _format_prompt(
        self,
        ref_codes: Any,
        ref_text: str,
        input_text: str,
        ref_phonemes: Optional[str] = None,
        input_phonemes: Optional[str] = None,
        use_chat_format: bool = False,
        emotion_tag: Optional[str] = None
    ) -> str:
        """
        Format the prompt for the TTS model.
        Common implementation for LMDeploy (Fast) and Remote backends.
        Standard backend uses a specialized chat template via tokenizer.

        Args:
            use_chat_format: If True, wraps the prompt with chat-style user/assistant
                             tokens (used by VieNeu-TTS GPU model). If False (default),
                             returns a compact prompt without those wrappers.
        """
        ref_codes_list = self.to_list(ref_codes)

        # Import inside method to avoid potential circular dependencies between
        # base TTS and phonemization utilities.
        from vieneu_utils.phonemize_text import phonemize_with_dict

        ref_text_phones = ref_phonemes if ref_phonemes else self.get_ref_phonemes(ref_text)
        input_text_phones = input_phonemes if input_phonemes else phonemize_with_dict(input_text, skip_normalize=True)
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes_list])

        emotion_prefix = emotion_tag if emotion_tag else ""

        if use_chat_format:
            return (
                f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{emotion_prefix}{ref_text_phones} {input_text_phones}"
                f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
            )
        return (
            f"<|TEXT_PROMPT_START|>{emotion_prefix}{ref_text_phones} {input_text_phones}"
            f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{codes_str}"
        )

    @abstractmethod
    def infer(self, text: str, apply_watermark: bool = True, **kwargs: Any) -> np.ndarray:
        """Main inference method for single text."""
        pass

    @abstractmethod
    def infer_batch(self, texts: List[str], apply_watermark: bool = True, **kwargs: Any) -> List[np.ndarray]:
        """Main inference method for batch processing."""
        pass

    def close(self) -> None:
        """Release resources."""
        pass

    def __enter__(self) -> 'BaseVieneuTTS':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
