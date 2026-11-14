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

            download_kwargs = {
                "repo_id": clean_handle,
                "local_dir": str(local_dir) if local_dir else None,
                "repo_type": "model",
            }
            if revision is not None:
                download_kwargs["revision"] = revision
            downloaded = snapshot_download(**download_kwargs)
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


def _is_mtp_key(name: str) -> bool:
    return name == "mtp" or name.startswith(("mtp.", "model.mtp."))


def _is_vision_key(name: str) -> bool:
    return name.startswith(("model.visual.", "visual.", "model.vision_tower.", "vision_tower."))


def _load_source_safetensors(source_dir: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file as st_load_file
    except ImportError as exc:
        raise ModelSourceError(f"safetensors.torch is required to read weights: {exc}") from exc

    index_path = source_dir / "model.safetensors.index.json"
    single_path = source_dir / "model.safetensors"
    tensors: dict[str, Any] = {}

    if index_path.is_file():
        try:
            index_obj = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelSourceError(f"invalid model.safetensors.index.json at {index_path}: {exc}") from exc
        weight_map = index_obj.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelSourceError(f"model.safetensors.index.json has empty weight_map at {index_path}")
        shard_names = sorted(set(str(v) for v in weight_map.values()))
        for shard_name in shard_names:
            shard_path = (source_dir / shard_name).resolve()
            if not shard_path.is_relative_to(source_dir.resolve()):
                raise ModelSourceError(f"shard path escapes source_dir: {shard_name!r}")
            if not shard_path.is_file():
                raise ModelSourceError(f"missing safetensors shard referenced by index: {shard_path}")
            shard_tensors = st_load_file(str(shard_path), device="cpu")
            for k, v in shard_tensors.items():
                if k in tensors:
                    raise ModelSourceError(f"duplicate tensor key across shards: {k!r}")
                tensors[k] = v
        missing_keys = set(weight_map.keys()) - set(tensors.keys())
        if missing_keys:
            raise ModelSourceError(
                f"shards missing indexed tensors: {sorted(missing_keys)[:5]}"
            )
        return tensors

    if single_path.is_file():
        return st_load_file(str(single_path), device="cpu")

    raise ModelSourceError(
        f"no model.safetensors or model.safetensors.index.json found in {source_dir}"
    )


def _save_target_safetensors(
    tensors: dict[str, Any],
    dest_dir: Path,
    *,
    shard_size_mb: int = 2048,
    metadata: dict[str, str] | None = None,
) -> list[str]:
    import os

    import torch
    from safetensors.torch import save_file as st_save_file

    max_bytes = max(1, int(shard_size_mb)) * 1024 * 1024
    seen_ptrs: set[tuple[int, int, tuple[int, ...]]] = set()
    prepared: list[tuple[str, torch.Tensor, int]] = []
    for name in sorted(tensors.keys()):
        t = tensors[name].detach().cpu().contiguous()
        sig = (t.data_ptr(), t.numel() * t.element_size(), tuple(t.shape))
        if sig in seen_ptrs:
            t = t.clone()
        else:
            seen_ptrs.add(sig)
        nbytes = int(t.numel()) * int(t.element_size())
        prepared.append((name, t, nbytes))

    if not prepared:
        return []

    shards: list[list[tuple[str, torch.Tensor]]] = [[]]
    cur_bytes = 0
    for name, t, nbytes in prepared:
        if shards[-1] and cur_bytes + nbytes > max_bytes:
            shards.append([(name, t)])
            cur_bytes = nbytes
        else:
            shards[-1].append((name, t))
            cur_bytes += nbytes

    str_meta = {str(k): str(v) for k, v in metadata.items()} if metadata else None

    if len(shards) == 1:
        shard_name = "model.safetensors"
        out_path = dest_dir / shard_name
        tmp_path = dest_dir / (shard_name + ".tmp")
        st_save_file({n: t for n, t in shards[0]}, str(tmp_path), metadata=str_meta)
        os.replace(tmp_path, out_path)
        return [shard_name]

    total = len(shards)
    weight_map: dict[str, str] = {}
    shard_files: list[str] = []
    total_size = sum(nb for _, _, nb in prepared)
    for idx, items in enumerate(shards, start=1):
        shard_name = f"model-{idx:05d}-of-{total:05d}.safetensors"
        out_path = dest_dir / shard_name
        tmp_path = dest_dir / (shard_name + ".tmp")
        st_save_file({n: t for n, t in items}, str(tmp_path), metadata=str_meta)
        os.replace(tmp_path, out_path)
        shard_files.append(shard_name)
        for n, _ in items:
            weight_map[n] = shard_name

    index_obj = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_tmp = dest_dir / "model.safetensors.index.json.tmp"
    index_tmp.write_text(
        json.dumps(index_obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(index_tmp, dest_dir / "model.safetensors.index.json")
    return shard_files


def _build_causal_text_config(src_config: dict[str, Any]) -> dict[str, Any]:
    import copy

    raw_text = src_config.get("text_config")
    if isinstance(raw_text, dict):
        cfg = copy.deepcopy(raw_text)
    else:
        cfg = copy.deepcopy(src_config)

    for vision_key in (
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        cfg.pop(vision_key, None)

    src_model_type = str(src_config.get("model_type", ""))
    cfg_model_type = str(cfg.get("model_type", src_model_type or "qwen3_5_text"))
    if src_model_type == "qwen3_5" or cfg_model_type == "qwen3_5":
        cfg["model_type"] = "qwen3_5_text"
    else:
        cfg["model_type"] = cfg_model_type

    archs = cfg.get("architectures") or src_config.get("architectures") or []
    if cfg["model_type"] == "qwen3_5_text":
        cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    elif archs:
        cfg["architectures"] = [
            a.replace("ForConditionalGeneration", "ForCausalLM") for a in archs
        ]

    for shared_key in (
        "tie_word_embeddings",
        "dtype",
        "torch_dtype",
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
        "transformers_version",
    ):
        if shared_key not in cfg and shared_key in src_config:
            cfg[shared_key] = copy.deepcopy(src_config[shared_key])
    return cfg


def extract_text_causal_lm_checkpoint(
    source_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    text_output_dir: Path | str | None = None,
    vision_output_dir: Path | str | None = None,
    shard_size_mb: int = 2048,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract a standalone causal text BF16 checkpoint from multimodal Ornith.

    Remaps ``model.language_model.* -> model.*``, keeps ``lm_head.*``, separates
    ``model.visual.*`` into ``vision_sidecar/``, excludes ``mtp.*`` keys, and
    promotes ``text_config`` to top-level ``config.json`` compatible with
    ``AutoModelForCausalLM``.

    # Errors: raises :class:`ModelSourceError` when weights/config are missing,
    keys cannot be routed cleanly, or destination exists without ``overwrite=True``.
    """
    import copy
    import os
    import shutil

    src = Path(source_dir).resolve()
    if not src.is_dir():
        raise ModelSourceError(f"source_dir does not exist: {src}")
    src_config = load_config(src)
    is_multimodal = isinstance(src_config.get("text_config"), dict) or (
        "vision_config" in src_config
    ) or str(src_config.get("model_type", "")) == "qwen3_5"

    if output_dir is not None:
        base = Path(output_dir).resolve()
        text_dir = (
            Path(text_output_dir).resolve()
            if text_output_dir is not None
            else (base / "teacher_text_bf16")
        )
        vision_dir = (
            Path(vision_output_dir).resolve()
            if vision_output_dir is not None
            else (base / "vision_sidecar")
        )
    elif text_output_dir is not None:
        text_dir = Path(text_output_dir).resolve()
        vision_dir = (
            Path(vision_output_dir).resolve()
            if vision_output_dir is not None
            else (text_dir.parent / "vision_sidecar")
        )
    else:
        raise ModelSourceError(
            "extract_text_causal_lm_checkpoint requires output_dir or text_output_dir"
        )

    for target in (text_dir, vision_dir):
        if target.exists() and any(target.iterdir()) and not overwrite:
            raise ModelSourceError(
                f"destination directory already exists and overwrite=False: {target}"
            )

    all_tensors = _load_source_safetensors(src)
    if not all_tensors:
        raise ModelSourceError(f"source checkpoint has no tensors: {src}")

    text_tensors: dict[str, Any] = {}
    vision_tensors: dict[str, Any] = {}
    excluded_mtp_keys: list[str] = []
    remapped_prefix_count = 0

    for name in sorted(all_tensors.keys()):
        tensor = all_tensors[name]
        if _is_mtp_key(name):
            excluded_mtp_keys.append(name)
            continue
        if _is_vision_key(name):
            vision_tensors[name] = tensor
            continue
        if name.startswith("model.language_model."):
            suffix = name[len("model.language_model.") :]
            if suffix.startswith("lm_head."):
                new_name = suffix
            else:
                new_name = f"model.{suffix}"
            remapped_prefix_count += 1
        elif name.startswith("language_model."):
            suffix = name[len("language_model.") :]
            new_name = suffix if suffix.startswith("lm_head.") else f"model.{suffix}"
            remapped_prefix_count += 1
        elif name.startswith("lm_head."):
            new_name = name
        elif name.startswith("model.lm_head."):
            new_name = name[len("model.") :]
        elif not is_multimodal and name.startswith("model."):
            new_name = name
        else:
            raise ModelSourceError(
                f"unclassified checkpoint tensor key in multimodal extraction: {name!r}"
            )

        if new_name in text_tensors:
            raise ModelSourceError(f"key collision after prefix remap: {new_name!r}")
        text_tensors[new_name] = tensor

    if not text_tensors:
        raise ModelSourceError(f"no text tensors extracted from {src}")

    text_cfg = _build_causal_text_config(src_config)
    vision_cfg = {
        k: copy.deepcopy(src_config[k])
        for k in (
            "model_type",
            "architectures",
            "vision_config",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        )
        if k in src_config
    }

    text_dir.parent.mkdir(parents=True, exist_ok=True)
    vision_dir.parent.mkdir(parents=True, exist_ok=True)
    text_tmp = text_dir.with_name(text_dir.name + ".incomplete")
    vision_tmp = vision_dir.with_name(vision_dir.name + ".incomplete")
    for tmp in (text_tmp, vision_tmp):
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

    vision_only_files = {
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "image_processor_config.json",
    }
    ignored_files = {
        "config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    ignored_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".onnx")

    copied_assets: list[str] = []
    vision_assets: list[str] = []

    try:
        (text_tmp / "config.json").write_text(
            json.dumps(text_cfg, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        text_shards = _save_target_safetensors(
            text_tensors,
            text_tmp,
            shard_size_mb=shard_size_mb,
            metadata={"format": "pt", "extracted_from": str(src.name)},
        )

        vision_shards: list[str] = []
        if vision_tensors:
            vision_shards = _save_target_safetensors(
                vision_tensors,
                vision_tmp,
                shard_size_mb=shard_size_mb,
                metadata={"format": "pt", "sidecar_from": str(src.name)},
            )
        if vision_cfg:
            (vision_tmp / "config.json").write_text(
                json.dumps(vision_cfg, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        for entry in sorted(src.iterdir()):
            if not entry.is_file():
                continue
            fname = entry.name
            if fname in ignored_files or fname.endswith(ignored_suffixes):
                continue
            if fname in vision_only_files:
                shutil.copy2(str(entry), str(vision_tmp / fname))
                vision_assets.append(fname)
            else:
                shutil.copy2(str(entry), str(text_tmp / fname))
                copied_assets.append(fname)

        summary = {
            "source_dir": str(src),
            "text_dir": str(text_dir),
            "vision_dir": str(vision_dir),
            "source_tensor_count": len(all_tensors),
            "text_tensor_count": len(text_tensors),
            "vision_tensor_count": len(vision_tensors),
            "excluded_mtp_count": len(excluded_mtp_keys),
            "remapped_prefix_count": remapped_prefix_count,
            "excluded_mtp_keys": excluded_mtp_keys,
            "text_keys": sorted(text_tensors.keys()),
            "vision_keys": sorted(vision_tensors.keys()),
            "text_model_type": text_cfg.get("model_type"),
            "text_architectures": list(text_cfg.get("architectures", [])),
            "tie_word_embeddings": text_cfg.get("tie_word_embeddings"),
            "text_shards": text_shards,
            "vision_shards": vision_shards,
            "copied_assets": copied_assets,
            "vision_assets": vision_assets,
        }
        (vision_tmp / "vision_sidecar_manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if text_dir.exists() and overwrite:
            shutil.rmtree(text_dir)
        os.replace(text_tmp, text_dir)

        if vision_dir.exists() and overwrite:
            shutil.rmtree(vision_dir)
        os.replace(vision_tmp, vision_dir)
        return summary
    except Exception:
        for tmp in (text_tmp, vision_tmp):
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        raise


def load_text_causal_model(
    model_dir: Path | str,
    device: str = "cuda:0",
    dtype: object = None,
    low_cpu_mem_usage: bool = True,
    *,
    torch_dtype: object = None,
    trust_remote_code: bool = False,
) -> object:
    """Load an extracted standalone text-only causal LM (``AutoModelForCausalLM``).

    Rejects unextracted multimodal checkpoints rather than silently misloading.

    # Errors: raises :class:`ModelSourceError` when the checkpoint is multimodal,
    missing ``config.json``, or transformers fails to load it.
    """
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise ModelSourceError(
            f"torch and transformers are required for causal model loading: {exc}"
        ) from exc

    resolved_dir = Path(model_dir).resolve()
    raw_cfg = load_config(resolved_dir)
    if (
        "vision_config" in raw_cfg
        or str(raw_cfg.get("model_type", "")) == "qwen3_5"
        or "Qwen3_5ForConditionalGeneration" in list(raw_cfg.get("architectures") or [])
    ):
        raise ModelSourceError(
            f"{resolved_dir} is a multimodal checkpoint; run extract_text_causal_lm_checkpoint() first"
        )

    effective_dtype = dtype if dtype is not None else torch_dtype
    if effective_dtype is None:
        effective_dtype = torch.bfloat16
    elif isinstance(effective_dtype, str):
        effective_dtype = getattr(torch, effective_dtype)

    try:
        cfg = AutoConfig.from_pretrained(str(resolved_dir), trust_remote_code=trust_remote_code)
        kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "trust_remote_code": trust_remote_code,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(resolved_dir), config=cfg, dtype=effective_dtype, **kwargs
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                str(resolved_dir), config=cfg, torch_dtype=effective_dtype, **kwargs
            )
        if device is not None and str(device) != "cpu":
            model = model.to(device)
        if not any(b.is_meta for b in model.buffers()):
            from .loader import restore_qwen3_5_nonpersistent_buffers

            restore_qwen3_5_nonpersistent_buffers(model)
        return model
    except Exception as exc:
        raise ModelSourceError(
            f"failed to load causal text model from {resolved_dir}: {type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "DEFAULT_MODEL_HANDLE",
    "HF_MODEL_HANDLE",
    "KAGGLE_MODEL_HANDLE",
    "import_model",
    "load_config",
    "load_foundation_model",
    "load_processor_and_tokenizer",
    "extract_text_causal_lm_checkpoint",
    "load_text_causal_model",
    "discover_architecture",
    "freeze_targets",
    "count_trainable",
    "ArchitectureReport",
    "LayerSummary",
]
