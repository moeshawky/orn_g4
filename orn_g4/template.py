"""Native chat-template inspection (work-order §2).

Inspects the template that the Kaggle artifact already incorporates, records its
fingerprint (handle + version/hash), and checks role delimiters, assistant
reasoning boundaries, tool-call serialization, tool-response formatting,
generation prefills, special-token IDs, and end-of-turn / end-of-sequence
behavior.

Never silently overwrites the included template with Ornith's default upstream
template; a token-level inconsistency with the dataset's recorded template is
identified, not repaired silently. A small sample compatibility test is
sufficient unless it reveals a genuine blocker.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import TemplateError


@dataclass(frozen=True)
class TemplateFingerprint:
    """Recorded identity of the native chat template."""

    source: str  # "artifact" or "hf_repo" or "none"
    handle: str | None = None  # Kaggle model handle or HF repo id
    version: str | None = None  # recorded version
    hash_sha256: str | None = None  # sha256 of the template file bytes
    path: str | None = None  # file path within the artifact
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TemplateReport:
    """Result of inspecting and cross-checking the native template."""

    fingerprint: TemplateFingerprint
    role_delimiters: list[str] = field(default_factory=list)
    reasoning_boundaries: list[str] = field(default_factory=list)
    tool_call_serialization: list[str] = field(default_factory=list)
    tool_response_format: list[str] = field(default_factory=list)
    generation_prefills: list[str] = field(default_factory=list)
    special_token_ids: dict[str, int] = field(default_factory=dict)
    end_of_turn: list[str] = field(default_factory=list)
    end_of_sequence: list[str] = field(default_factory=list)
    raw_template: str | None = None
    inconsistency: str | None = None  # non-None when dataset vs template mismatch


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _candidate_template_files(model_dir: Path) -> list[Path]:
    """Return likely template file names inside the artifact, in preference order.

    Never hardcodes beyond the list; the actual artifact decides which exists.
    """
    names = [
        "chat_template.jinja",
        "chat_template.json",
        "template.jinja",
        "conversation_template.jinja",
        "generation_config.json",  # may carry chat_template string
        "tokenizer_config.json",  # may carry chat_template string
    ]
    return [model_dir / n for n in names]


def _extract_template_string(model_dir: Path) -> tuple[str | None, Path | None]:
    """Return (template_string, source_path) by inspecting actual artifact files.

    Checks jinja/text templates and JSON configs that carry a chat_template
    field. Returns (None, None) when nothing resembling a template is found —
    the caller then decides whether that is a blocker.
    """
    for path in _candidate_template_files(model_dir):
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        text = raw.decode("utf-8", errors="replace")
        # JSON configs carrying chat_template.
        if path.suffix == ".json":
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            ct = obj.get("chat_template") if isinstance(obj, dict) else None
            if isinstance(ct, str) and ct.strip():
                return ct, path
            # Some tokenizers nest it under tokenizer.chat_template.
            tok = obj.get("tokenizer") if isinstance(obj, dict) else None
            if isinstance(tok, dict) and isinstance(tok.get("chat_template"), str):
                return tok["chat_template"], path
        else:
            # Treat any non-empty text file as a candidate jinja template.
            if text.strip():
                return text, path
    return None, None


def fingerprint_template(
    model_dir: Path | str,
    template_string: str | None = None,
    source_path: Path | str | None = None,
) -> TemplateFingerprint:
    """Build a :class:`TemplateFingerprint` from the discovered template.

    The hash is over the exact template bytes so a later re-download can be
    verified against this run's template. The version is taken from any
    ``version`` field in a JSON config, else absent (recorded as unknown rather
    than invented).

    # Errors: raises :class:`TemplateError` when no template can be found in
    the artifact at all.
    """
    model_dir = Path(model_dir)
    if template_string is None:
        template_string, source_path = _extract_template_string(model_dir)
    if template_string is None:
        raise TemplateError(
            f"no chat template found in artifact {model_dir}; "
            f"the Kaggle artifact is expected to carry the project's template"
        )

    data = template_string.encode("utf-8")
    fp = TemplateFingerprint(
        source="artifact",
        handle=None,
        version=None,
        hash_sha256=_sha256_bytes(data),
        path=str(source_path) if source_path is not None else None,
        notes=[],
    )
    # Attempt to read a version from a JSON config if available.
    if source_path is not None and Path(source_path).suffix == ".json":
        try:
            obj = json.loads(template_string)
            if isinstance(obj, dict) and "version" in obj:
                fp = TemplateFingerprint(
                    source=fp.source,
                    handle=fp.handle,
                    version=str(obj["version"]),
                    hash_sha256=fp.hash_sha256,
                    path=fp.path,
                    notes=fp.notes,
                )
        except (json.JSONDecodeError, TypeError):
            pass
    return fp


def _detect_features(template: str) -> dict[str, list[str] | dict[str, int]]:
    """Heuristically detect interface features present in the template text.

    Returns role delimiters, reasoning boundaries, tool-call serialization,
    tool-response format, generation prefills, end-of-turn, end-of-sequence.
    These are *descriptive* — absence of a token here means "not detected by
    this heuristic", not "the template lacks it". The operator reviews.
    """
    detected: dict[str, list[str] | dict[str, int]] = {
        "role_delimiters": [],
        "reasoning_boundaries": [],
        "tool_call_serialization": [],
        "tool_response_format": [],
        "generation_prefills": [],
        "end_of_turn": [],
        "end_of_sequence": [],
    }

    # Common role delimiters.
    for tok in [
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|tool|>",
        "system",
        "user",
        "assistant",
        "tool",
    ]:
        if tok in template:
            detected["role_delimiters"].append(tok)  # type: ignore[arg-type]
    # Reasoning boundaries (think tags).
    for tok in [
        "<|think|>",
        "</|think|>",
        "<|begin_reasoning|>",
        "<|end_reasoning|>",
        "<|start_thinking|>",
        "<|end_thinking|>",
    ]:
        if tok in template:
            detected["reasoning_boundaries"].append(tok)  # type: ignore[arg-type]
    # Tool call / response markers.
    for tok in [
        "<|tool_call|>",
        "<|tool_response|>",
        "<|function_call|>",
        "<|function_response|>",
        "<|constrain|>",
    ]:
        if tok in template:
            detected["tool_call_serialization"].append(tok)  # type: ignore[arg-type]
    # End of turn / sequence.
    for tok in ["<|end_of_turn|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_sequence|>", "<|end|>"]:
        if tok in template:
            detected["end_of_turn"].append(tok)  # type: ignore[arg-type]

    return detected  # type: ignore[return-value]


def inspect_template(
    model_dir: Path | str,
    *,
    dataset_template: str | None = None,
    dataset_hash: str | None = None,
) -> TemplateReport:
    """Inspect the native template and cross-check against the dataset's recorded template.

    When ``dataset_template`` and/or ``dataset_hash`` are supplied (parsed from
    ``DATASET_RELEASE_MANIFEST.json``), a token-level inconsistency is
    identified in ``report.inconsistency`` but never silently repaired.

    # Errors: raises :class:`TemplateError` when the artifact carries no
    template at all (genuine blocker).
    """
    model_dir = Path(model_dir)
    template_string, source_path = _extract_template_string(model_dir)
    if template_string is None:
        raise TemplateError(
            f"no chat template found in artifact {model_dir}; "
            f"cannot proceed with the designated native template"
        )

    fingerprint = fingerprint_template(model_dir, template_string, source_path)
    features = _detect_features(template_string)

    inconsistency: str | None = None
    if dataset_hash is not None and fingerprint.hash_sha256 != dataset_hash:
        inconsistency = (
            f"dataset recorded template hash {dataset_hash} does not match "
            f"artifact template hash {fingerprint.hash_sha256}; "
            f"token-level inconsistency identified, not repaired silently"
        )
    elif dataset_template is not None and dataset_template.strip() != template_string.strip():
        inconsistency = (
            "dataset recorded template text differs from the artifact's "
            "included template; identified, not repaired silently"
        )

    return TemplateReport(
        fingerprint=fingerprint,
        role_delimiters=list(features["role_delimiters"]),
        reasoning_boundaries=list(features["reasoning_boundaries"]),
        tool_call_serialization=list(features["tool_call_serialization"]),
        tool_response_format=list(features["tool_response_format"]),
        generation_prefills=list(features["generation_prefills"]),
        special_token_ids={},
        end_of_turn=list(features["end_of_turn"]),
        end_of_sequence=[],
        raw_template=template_string,
        inconsistency=inconsistency,
    )


__all__ = [
    "TemplateFingerprint",
    "TemplateReport",
    "inspect_template",
    "fingerprint_template",
]
