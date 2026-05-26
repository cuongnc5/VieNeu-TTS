from pathlib import Path
from typing import Optional, Union, List, Generator, Any, Dict
import numpy as np
import torch
import gc
import logging
from collections import defaultdict
from .base import BaseVieneuTTS
from .utils import _compile_codec_with_triton, extract_speech_ids, _linear_overlap_add, normalize_device
from vieneu_utils.phonemize_text import phonemize_batch
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks

logger = logging.getLogger("Vieneu.Fast")

class FastVieNeuTTS(BaseVieneuTTS):
    """
    GPU-optimized VieNeu-TTS using LMDeploy TurbomindEngine.
    """

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS",
        backbone_device: str = "cuda",
        codec_repo: str = "neuphonic/distill-neucodec",
        codec_device: str = "cuda",
        memory_util: float = 0.3,
        tp: int = 1,
        enable_prefix_caching: bool = False,
        quant_policy: int = 0,
        enable_triton: bool = True,
        max_batch_size: int = 4,
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.device = backbone_device

        if backbone_device != "cuda" and not backbone_device.startswith("cuda:"):
            raise ValueError("LMDeploy backend requires CUDA device")

        # Streaming configuration
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 50
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        self.max_batch_size = max_batch_size
        self._ref_cache: Dict[str, Any] = {}
        self.stored_dict = defaultdict(dict)

        self._is_onnx_codec = False
        self._triton_enabled = False

        self.use_chat_format = backbone_repo.rstrip("/").endswith("pnnbao-ump/VieNeu-TTS")

        self._load_backbone_lmdeploy(backbone_repo, memory_util, tp, enable_prefix_caching, quant_policy, hf_token)
        self._load_codec(codec_repo, codec_device, enable_triton)
        self._load_voices(backbone_repo, hf_token)
        self._warmup_model()

        logger.info("✅ FastVieNeuTTS with optimizations loaded successfully!")
        logger.info(f"   Max batch size: {self.max_batch_size}")

    def _load_backbone_lmdeploy(self, repo, memory_util, tp, enable_prefix_caching, quant_policy, hf_token=None):
        logger.info(f"Loading backbone with LMDeploy from: {repo}")
        if hf_token:
            import os
            os.environ["HF_TOKEN"] = hf_token

        try:
            from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
        except ImportError as e:
            raise ImportError(
                "Failed to import `lmdeploy`. Install with: pip install vieneu[gpu]"
            ) from e

        backend_config = TurbomindEngineConfig(
            cache_max_entry_count=memory_util,
            tp=tp,
            enable_prefix_caching=enable_prefix_caching,
            dtype='bfloat16',
            quant_policy=quant_policy
        )
        self.backbone = pipeline(repo, backend_config=backend_config)
        self.gen_config = GenerationConfig(
            top_p=0.95, top_k=50, temperature=1.0, max_new_tokens=2048,
            repetition_penalty=1.2,
            do_sample=True, min_new_tokens=40,
        )

    def _load_codec(self, codec_repo: str, codec_device: str, enable_triton: bool) -> None:
        super()._load_codec(codec_repo, codec_device)

        if enable_triton and not getattr(self, "_is_onnx_codec", False) and codec_device != "cpu":
            self._triton_enabled = _compile_codec_with_triton(self.codec)

    def _warmup_model(self):
        logger.info("🔥 Warming up model...")
        try:
            dummy_codes = list(range(10))
            dummy_prompt = self._format_prompt(dummy_codes, "warmup", "test", use_chat_format=self.use_chat_format)
            _ = self.backbone([dummy_prompt], gen_config=self.gen_config, do_preprocess=False)
            logger.info("   ✅ Warmup complete")
        except Exception as e:
            logger.warning(f"   ⚠️ Warmup failed: {e}")

    def _decode(self, codes_str: str) -> np.ndarray:
        speech_ids = extract_speech_ids(codes_str)
        if not speech_ids:
            raise ValueError(
                "No valid speech tokens found in the output. "
                "Lỗi này có thể do GPU của bạn không hỗ trợ định dạng bfloat16 (ví dụ: dòng T4, RTX 20-series) "
                "dẫn đến sai số khi tính toán. Bạn hãy thử chuyển sang dùng phiên bản VieNeu-TTS-0.3B nếu vẫn muốn dùng LmDeploy hoặc "
                "bỏ chọn 'LMDeploy' trong Tùy chọn nâng cao. Nếu vẫn gặp lỗi này, hãy thông báo với chúng tôi tại: https://discord.com/invite/yJt8kzjzWZ"
            )

        if self._is_onnx_codec:
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes)
        else:
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(self.codec.device)
                recon = self.codec.decode_code(codes).cpu().numpy()
        return recon[0, 0, :]


    def infer(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, silence_p: float = 0.15, crossfade_p: float = 0.0, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True, **kwargs) -> np.ndarray:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)
        original_text = text
        text, _ = self._prepare_text_for_inference(
            logger,
            text,
            rewrite_enabled=kwargs.get("rewrite_problematic_short_text", True),
        )

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        self.gen_config.temperature = temperature
        self.gen_config.top_k = top_k

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        self._log_text_preparation(logger, original_text, text, chunks)
        if not chunks:
            return np.array([], dtype=np.float32)

        if len(chunks) == 1:
            ref_phonemes = self.get_ref_phonemes(ref_text)
            self._log_reference_context(logger, ref_text, ref_phonemes, ref_codes)
            chunk_phonemes = phonemize_batch(chunks, skip_normalize=True)
            guard = self._build_generation_guard(
                chunks[0],
                chunk_phonemes[0],
                target_duration_s=kwargs.get("target_duration"),
                requested_max_new_tokens=kwargs.get("max_new_tokens"),
            )
            sampling = self._build_sampling_controls(
                temperature=temperature,
                top_k=top_k,
                guard=guard,
                repetition_penalty=self.gen_config.repetition_penalty,
            )
            self.gen_config.max_new_tokens = guard["max_new_tokens"]
            self.gen_config.min_new_tokens = guard["min_new_tokens"]
            self.gen_config.temperature = sampling["temperature"]
            self.gen_config.top_k = sampling["top_k"]
            self.gen_config.repetition_penalty = sampling["repetition_penalty"]
            self._log_chunk_inputs(
                logger,
                chunk_index=0,
                total_chunks=1,
                chunk_text=chunks[0],
                chunk_phonemes=chunk_phonemes[0],
                extra={
                    "guard_max_tokens": guard["max_new_tokens"],
                    "guard_short_text": guard["short_text"],
                    "guard_compact": guard["repetitive_compact"],
                    "guard_max_sec": guard["desired_max_duration_s"],
                    "guard_temp": sampling["temperature"],
                    "guard_top_k": sampling["top_k"],
                },
            )
            prompt = self._format_prompt(
                ref_codes,
                ref_text,
                chunks[0],
                ref_phonemes=ref_phonemes,
                input_phonemes=chunk_phonemes[0],
                use_chat_format=self.use_chat_format,
                emotion_tag=kwargs.get('emotion_tag'),
            )
            responses = self.backbone([prompt], gen_config=self.gen_config, do_preprocess=False)
            output_str = self._apply_generation_guard(
                logger,
                chunk_text=chunks[0],
                chunk_phonemes=chunk_phonemes[0],
                output_str=responses[0].text,
                guard=guard,
            )
            wav = self._decode(output_str)
            self._log_generation_result(
                logger,
                chunk_index=0,
                total_chunks=1,
                chunk_text=chunks[0],
                chunk_phonemes=chunk_phonemes[0],
                output_str=output_str,
                wav=wav,
            )
            if apply_watermark:
                wav = self._apply_watermark(wav)
        else:
            all_wavs = self.infer_batch(
                chunks,
                ref_codes=ref_codes,
                ref_text=ref_text,
                voice=voice,
                temperature=temperature,
                top_k=top_k,
                skip_normalize=True,
                apply_watermark=False,
                **kwargs,
            )
            wav = join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)
            if apply_watermark:
                wav = self._apply_watermark(wav)

        return wav

    def infer_batch(self, texts: List[str], ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True, max_batch_size: Optional[int] = None, **kwargs) -> List[np.ndarray]:
        texts = self._prepare_texts_for_inference(
            logger,
            texts,
            rewrite_enabled=kwargs.get("rewrite_problematic_short_text", True),
        )

        if not skip_normalize:
            texts = [self.normalizer.normalize(t) for t in texts]

        max_batch_size = max_batch_size or self.max_batch_size

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        # Pre-phonemize all for performance
        ref_phonemes = self.get_ref_phonemes(ref_text)
        self._log_reference_context(logger, ref_text, ref_phonemes, ref_codes)
        chunk_phonemes = phonemize_batch(texts, skip_normalize=True)
        guards = [
            self._build_generation_guard(
                text,
                phonemes,
                requested_max_new_tokens=kwargs.get("max_new_tokens"),
            )
            for text, phonemes in zip(texts, chunk_phonemes)
        ]
        samplings = [
            self._build_sampling_controls(
                temperature=temperature,
                top_k=top_k,
                guard=guard,
                repetition_penalty=self.gen_config.repetition_penalty,
            )
            for guard in guards
        ]
        for i, (text, phonemes) in enumerate(zip(texts, chunk_phonemes)):
            self._log_chunk_inputs(
                logger,
                chunk_index=i,
                total_chunks=len(texts),
                chunk_text=text,
                chunk_phonemes=phonemes,
                extra={
                    "guard_max_tokens": guards[i]["max_new_tokens"],
                    "guard_short_text": guards[i]["short_text"],
                    "guard_compact": guards[i]["repetitive_compact"],
                    "guard_max_sec": guards[i]["desired_max_duration_s"],
                    "guard_temp": samplings[i]["temperature"],
                    "guard_top_k": samplings[i]["top_k"],
                },
            )

        self.gen_config.temperature = min(sampling["temperature"] for sampling in samplings)
        self.gen_config.top_k = min(sampling["top_k"] for sampling in samplings)
        self.gen_config.repetition_penalty = max(sampling["repetition_penalty"] for sampling in samplings)
        self.gen_config.max_new_tokens = max(guard["max_new_tokens"] for guard in guards)
        self.gen_config.min_new_tokens = min(guard["min_new_tokens"] for guard in guards)

        all_wavs = []
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i : i + max_batch_size]
            batch_phonemes = chunk_phonemes[i : i + max_batch_size]
            prompts = [self._format_prompt(ref_codes, ref_text, text, ref_phonemes=ref_phonemes, 
                                          input_phonemes=ph, use_chat_format=self.use_chat_format,
                                          emotion_tag=kwargs.get('emotion_tag'))
                      for text, ph in zip(batch_texts, batch_phonemes)]
            responses = self.backbone(prompts, gen_config=self.gen_config, do_preprocess=False)
            batch_codes = [response.text for response in responses]
            guarded_outputs = [
                self._apply_generation_guard(
                    logger,
                    chunk_text=chunk_text,
                    chunk_phonemes=chunk_phoneme,
                    output_str=output_str,
                    guard=guards[i + offset],
                )
                for offset, (chunk_text, chunk_phoneme, output_str) in enumerate(zip(batch_texts, batch_phonemes, batch_codes))
            ]
            batch_wavs = [self._decode(codes) for codes in guarded_outputs]
            for offset, (chunk_text, chunk_phoneme, output_str, wav) in enumerate(zip(batch_texts, batch_phonemes, guarded_outputs, batch_wavs)):
                self._log_generation_result(
                    logger,
                    chunk_index=i + offset,
                    total_chunks=len(texts),
                    chunk_text=chunk_text,
                    chunk_phonemes=chunk_phoneme,
                    output_str=output_str,
                    wav=wav,
                )
            if apply_watermark:
                batch_wavs = [self._apply_watermark(w) for w in batch_wavs]
            all_wavs.extend(batch_wavs)
        return all_wavs

    def infer_stream(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, **kwargs) -> Generator[np.ndarray, None, None]:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)
        text, _ = self._prepare_text_for_inference(
            logger,
            text,
            rewrite_enabled=kwargs.get("rewrite_problematic_short_text", True),
        )

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        self.gen_config.temperature = temperature
        self.gen_config.top_k = top_k

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        for chunk in chunks:
            yield from self._infer_stream_single(chunk, ref_codes, ref_text, emotion_tag=kwargs.get('emotion_tag'))

    def _infer_stream_single(self, text: str, ref_codes: Any, ref_text: str, emotion_tag: Optional[str] = None) -> Generator[np.ndarray, None, None]:
        ref_codes_list = self.to_list(ref_codes)
        prompt = self._format_prompt(ref_codes_list, ref_text, text, use_chat_format=self.use_chat_format, emotion_tag=emotion_tag)
        audio_cache = []
        token_cache = [f"<|speech_{idx}|>" for idx in ref_codes_list]
        n_decoded_samples = 0
        n_decoded_tokens = len(ref_codes_list)

        for response in self.backbone.stream_infer([prompt], gen_config=self.gen_config, do_preprocess=False):
            output_str = response.text
            new_tokens = output_str[len("".join(token_cache[len(ref_codes_list):])):] if len(token_cache) > len(ref_codes_list) else output_str
            if new_tokens:
                token_cache.append(new_tokens)

            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:
                tokens_start = max(n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames, 0)
                tokens_end = n_decoded_tokens + self.streaming_frames_per_chunk + self.streaming_lookforward + self.streaming_overlap_frames
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = sample_start + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = self._apply_watermark(recon)
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        remaining_tokens = len(token_cache) - n_decoded_tokens
        if remaining_tokens > 0:
            tokens_start = max(len(token_cache) - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens), 0)
            sample_start = (len(token_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = self._apply_watermark(recon)
            recon = recon[sample_start:]
            audio_cache.append(recon)
            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon

    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_optimization_stats(self) -> Dict[str, Any]:
        return {
            'triton_enabled': self._triton_enabled,
            'max_batch_size': self.max_batch_size,
            'cached_references': len(self._ref_cache),
            'active_sessions': len(self.stored_dict),
            'prefix_caching': False,
        }
