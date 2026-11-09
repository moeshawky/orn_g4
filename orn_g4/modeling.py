"""Model import and architecture discovery for Ornith-1.5-9b (work-order §1).

Imports the exact Kaggle artifact through KaggleHub into ``MODEL_DIR`` and
discovers the checkpoint's *actual* module tree — layer classes, counts,
parameter totals, and tensor names — rather than assuming a generic Qwen
layout. torch/transformers are imported lazily inside functions so this module
imports on any host.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_MODEL_HANDLE, HF_MODEL_HANDLE, KAGGLE_MODEL_HANDLE
from .errors import ModelSourceError
from .utils import class_name


@dataclass(frozen=True)
class LayerSummary:
    """One discovered module type and its aggregate stats."""

    module_class: str
    count: int
    params: int
    example_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArchitectureReport:
    """Full discovered architecture of the loaded checkpoint."""

    model_class: str
    total_params: int
    trainable_params: int
    hidden_size: int | None
    num_layers: int | None
    layers: list[LayerSummary] = field(default_factory=list)
    tensor_names: list[str] = field(default_factory=list)
    has_vision_encoder: bool = False
    has_mtp: bool = False
    embeddings_tied: bool | None = None
    config_path: str = ""
    config_raw: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def import_model(
    handle: str = DEFAULT_MODEL_HANDLE,
    local_dir: str | Path | None = None,
    revision: str | None = None,
) -> Path:
    """Download or resolve the foundation model from Hugging Face, Kaggle, or local path.

    Resolution strategy:
    1. Local filesystem: If `handle` is an existing path or directory containing `config.json`,
       returns `Path(handle).resolve()`.
    2. Hugging Face Hub: If `handle` is a standard Hugging Face repo ID (e.g. 'moeshawky/Ornith-1.5-9B'
       or prefixed with 'hf://'), downloads via `huggingface_hub.snapshot_download`.
    3. KaggleHub: If `handle` is a Kaggle artifact handle (e.g. contains '/transformers/' or starts
       with 'kaggle://'), downloads via `kagglehub.model_download`.
    """
    local_candidate = Path(handle)
    if local_candidate.exists():
        p = local_candidate.resolve()
        if (p / "config.json").is_file():
            return p

    clean_handle = handle.removeprefix("hf://").strip()
    is_kaggle = clean_handle.startswith("kaggle://") or clean_handle.count("/") > 1

    if is_kaggle and revision is not None:
        raise ModelSourceError("revision pinning applies to Hugging Face; use a versioned Kaggle handle")
    if not is_kaggle:
        try:
            from huggingface_hub import snapshot_download

            downloaded = snapshot_download(
                repo_id=clean_handle,
                revision=revision,
                local_dir=str(local_dir) if local_dir else None,
                repo_type="model",
            )
            p = Path(downloaded).resolve()
            if (p / "config.json").is_file():
                return p
        except Exception as exc:
            raise ModelSourceError(
                f"Hugging Face snapshot_download({clean_handle!r}) failed: {type(exc).__name__}: {exc}"
            ) from exc

    try:
        import kagglehub
    except ImportError as exc:
        raise ModelSourceError(
            f"kagglehub is not installed and Hugging Face resolution failed for {handle!r}."
        ) from exc

    kaggle_handle = clean_handle.removeprefix("kaggle://")
    try:
        downloaded = kagglehub.model_download(kaggle_handle)
    except Exception as exc:
        raise ModelSourceError(
            f"kagglehub.model_download({kaggle_handle!r}) failed: {type(exc).__name__}: {exc}"
        ) from exc

    model_dir = Path(downloaded).resolve()
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ModelSourceError(
            f"downloaded artifact at {model_dir} has no config.json; "
            f"this is not the expected Ornith checkpoint"
        )

    return model_dir


def load_config(model_dir: Path | str) -> dict:
    """Load and return config.json from the model directory.

    # Errors: raises :class:`ModelSourceError` if missing or not valid JSON.
    """
    config_path = Path(model_dir) / "config.json"
    try:
        with Path(config_path).open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise ModelSourceError(f"config.json not found at {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelSourceError(f"config.json at {config_path} is not valid JSON: {exc}") from exc


def discover_architecture(model_dir: Path | str) -> ArchitectureReport:
    """Load the checkpoint and report its actual architecture.

    Loads with torch + transformers using the checkpoint's declared class
    (from config.json ``architectures`` / ``auto_map``), walks every named
    module, and aggregates parameter counts by module class. Also checks for a
    vision encoder, an MTP component, and tied vs untied embeddings/LM head.

    # Errors: raises :class:`ModelSourceError` when torch/transformers are
    unavailable, the model class cannot be instantiated, or the state dict
    cannot be loaded.
    """
    try:
        import torch  # lazy
    except ImportError as exc:
        raise ModelSourceError(
            "torch and transformers are required to load the checkpoint; "
            f"install them in the notebook before cell 03. "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    model_dir = Path(model_dir)
    config = load_config(model_dir)

    architectures = config.get("architectures")
    auto_map = config.get("auto_map", {})
    model_class_name = None
    if isinstance(architectures, list) and architectures:
        model_class_name = architectures[0]
    elif isinstance(auto_map, dict) and auto_map.get("AutoModelForMultimodalLM"):
        model_class_name = auto_map["AutoModelForMultimodalLM"]
    elif isinstance(auto_map, dict) and auto_map.get("AutoModelForCausalLM"):
        model_class_name = auto_map["AutoModelForCausalLM"]

    if model_class_name is None:
        raise ModelSourceError(
            "config.json declares no `architectures` or `auto_map` model class"
        )

    try:
        model_class = getattr(
            __import__("transformers", fromlist=[model_class_name]),
            model_class_name,
        )
    except (ImportError, AttributeError) as exc:
        raise ModelSourceError(f"transformers does not expose {model_class_name!r}: {exc}") from exc

    try:
        model = model_class.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    except Exception as exc:
        raise ModelSourceError(
            f"failed to instantiate {model_class_name} from {model_dir}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    model.eval()

    layer_summaries: dict[str, LayerSummary] = {}
    tensor_names: list[str] = []
    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        cls = class_name(module)
        p = sum(p_.numel() for p_ in module.parameters(recurse=False))
        total_params += p
        trainable_params += sum(
            p_.numel() for p_ in module.parameters(recurse=False) if p_.requires_grad
        )
        if p > 0:
            summary = layer_summaries.get(cls)
            if summary is None:
                layer_summaries[cls] = LayerSummary(
                    module_class=cls, count=1, params=p, example_names=[name]
                )
            else:
                layer_summaries[cls] = LayerSummary(
                    module_class=cls,
                    count=summary.count + 1,
                    params=summary.params + p,
                    example_names=(summary.example_names + [name])[:3],
                )

    for name, _ in model.named_parameters():
        tensor_names.append(name)

    hidden_size = config.get("hidden_size")
    num_layers = None
    for key in ("num_hidden_layers", "num_layers", "n_layer"):
        if key in config:
            num_layers = config[key]
            break

    has_vision = any("vision" in n.lower() or "visual" in n.lower() for n in tensor_names)
    has_mtp = any("mtp" in n.lower() for n in tensor_names)

    embeddings_tied = None
    try:
        emb = model.get_input_embeddings()
        lm = model.get_output_embeddings()
        if emb is not None and lm is not None:
            embeddings_tied = emb.weight.data_ptr() == lm.weight.data_ptr()
    except Exception:
        embeddings_tied = None

    layers = sorted(layer_summaries.values(), key=lambda s: -s.params)

    return ArchitectureReport(
        model_class=model_class_name,
        total_params=total_params,
        trainable_params=trainable_params,
        hidden_size=int(hidden_size) if isinstance(hidden_size, int) else None,
        num_layers=int(num_layers) if isinstance(num_layers, int) else None,
        layers=layers,
        tensor_names=tensor_names,
        has_vision_encoder=has_vision,
        has_mtp=has_mtp,
        embeddings_tied=embeddings_tied,
        config_path=str((model_dir / "config.json").resolve()),
        config_raw=config,
        notes=[],
    )


def freeze_targets(model: object, targets: list[str]) -> dict[str, bool]:
    """Freeze (requires_grad=False) the named top-level submodules.

    ``targets`` are attribute paths on the model (e.g. ``["model.vision_tower"]``).
    Returns a mapping of target -> whether it was found and frozen.

    # Errors: raises :class:`ModelSourceError` if ``model`` has no parameters
    at all (not a real loaded model).
    """
    if not any(True for _ in model.parameters()):
        raise ModelSourceError("model has no parameters; cannot freeze targets")
    results: dict[str, bool] = {}
    for target in targets:
        obj = model
        found = True
        for part in target.split("."):
            if not hasattr(obj, part):
                found = False
                break
            obj = getattr(obj, part)
        if found and hasattr(obj, "parameters"):
            for p in obj.parameters():
                p.requires_grad = False
            results[target] = True
        else:
            results[target] = False
    return results


def count_trainable(model: object) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_foundation_model(
    model_dir: Path | str,
    device: str = "cuda:0",
    dtype: object = None,
    low_cpu_mem_usage: bool = True,
    torch_dtype: object = None,
) -> object:
    """Load native foundation model (supporting Qwen3.5 multimodal architecture)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM

    model_dir = Path(model_dir)
    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=False)
    if dtype is not None and torch_dtype is not None and dtype != torch_dtype:
        raise ValueError("conflicting dtype and torch_dtype arguments")
    target_dtype = dtype if dtype is not None else (torch_dtype if torch_dtype is not None else torch.bfloat16)
    kwargs: dict = {
        "torch_dtype": target_dtype,
        "trust_remote_code": False,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if str(device) != "cpu" and torch.cuda.is_available():
        kwargs["device_map"] = {"": str(device)}

    if cfg.model_type == "qwen3_5" or "Qwen3_5ForConditionalGeneration" in getattr(cfg, "architectures", []):
        return AutoModelForMultimodalLM.from_pretrained(str(model_dir), **kwargs)
    try:
        return AutoModelForMultimodalLM.from_pretrained(str(model_dir), **kwargs)
    except Exception:
        return AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)


def load_processor_and_tokenizer(model_dir: Path | str) -> tuple[object, object]:
    """Load native AutoProcessor and AutoTokenizer for Ornith / Qwen3.5."""
    from transformers import AutoProcessor, AutoTokenizer

    model_dir = Path(model_dir)
    processor = None
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=False)
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception:
        pass
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    return processor, tokenizer


__all__ = [
    "DEFAULT_MODEL_HANDLE",
    "HF_MODEL_HANDLE",
    "KAGGLE_MODEL_HANDLE",
    "import_model",
    "load_config",
    "load_foundation_model",
    "load_processor_and_tokenizer",
    "discover_architecture",
    "freeze_targets",
    "count_trainable",
    "ArchitectureReport",
    "LayerSummary",
]
