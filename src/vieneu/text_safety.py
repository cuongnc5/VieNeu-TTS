from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


NORMALIZED_OK_CUE_TOKENS = {
    "ok",
    "okay",
    "oke",
    "okey",
}
NORMALIZED_FILLER_CUE_TOKENS = {
    "a",
    "à",
    "á",
    "ạ",
    "uh",
    "uhm",
    "um",
    "ùm",
    "ừ",
    "ừm",
    "ờ",
    "ờm",
    "ơ",
}


@dataclass(frozen=True)
class ShortCueRewritePlan:
    original_text: str
    normalized_text: str
    rewritten_text: str
    applied: bool
    reason: str = ""


def normalize_short_text_for_safety(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\b(?:o|ô)\s+k[eê]\b", "ok", normalized)
    return normalized


def _infer_rewrite_suffix(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    if stripped.endswith("?"):
        return "?"
    if stripped.endswith("!"):
        return "!"
    if stripped.endswith(("...", "…", ".")):
        return "."
    return ""


def rewrite_problematic_short_cue(text: str) -> ShortCueRewritePlan:
    normalized_text = normalize_short_text_for_safety(text)
    tokens = normalized_text.split() if normalized_text else []
    if not tokens or len(tokens) > 4:
        return ShortCueRewritePlan(
            original_text=text,
            normalized_text=normalized_text,
            rewritten_text=text,
            applied=False,
        )

    token_set = set(tokens)
    only_ok_tokens = token_set.issubset(NORMALIZED_OK_CUE_TOKENS)
    only_filler_tokens = token_set.issubset(NORMALIZED_FILLER_CUE_TOKENS)
    only_safe_short_tokens = token_set.issubset(NORMALIZED_OK_CUE_TOKENS | NORMALIZED_FILLER_CUE_TOKENS)

    rewritten_core = ""
    reason = ""
    if only_ok_tokens:
        rewritten_core = "ok rồi"
        reason = "ok_only"
    elif only_filler_tokens:
        rewritten_core = "à vâng"
        reason = "filler_only"
    elif only_safe_short_tokens and tokens[0] in NORMALIZED_FILLER_CUE_TOKENS:
        rewritten_core = "à vâng"
        reason = "filler_led_mixed_short_phrase"
    elif only_safe_short_tokens and any(token in NORMALIZED_OK_CUE_TOKENS for token in tokens):
        rewritten_core = "ok rồi"
        reason = "ok_led_mixed_short_phrase"

    if not rewritten_core:
        return ShortCueRewritePlan(
            original_text=text,
            normalized_text=normalized_text,
            rewritten_text=text,
            applied=False,
        )

    suffix = _infer_rewrite_suffix(text)
    rewritten_text = rewritten_core.rstrip(" .!?")
    if suffix:
        rewritten_text = f"{rewritten_text}{suffix}"

    return ShortCueRewritePlan(
        original_text=text,
        normalized_text=normalized_text,
        rewritten_text=rewritten_text,
        applied=rewritten_text != text,
        reason=reason,
    )
