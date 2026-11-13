# AGENTS.md — orn_g4 Entry Gate

*Deterministic 1.58-bit ternary quantization, two-stage recovery, and canonical safetensors for Ornith-1.5-9B — what to run, where the traps are, and which names this repo's own docstrings get wrong.*

## THE IRON LAW

```
LAZY IMPORTS ARE LOAD-BEARING — torch/transformers/kagglehub/peft stay inside functions; the package must import on a CPU-only host.
DOCSTRINGS ARE NOT THE API — orn_g4/errors.py cites 8 function references across 6 modules that do not exist; resolve names against the module namespace.
ARTIFACTS ARE IMMUTABLE — never write into a checkpoint dir in place; the controller re-hashes on resume and re-runs a stage whose checksum moved.
THE REPO ROOT MUST BE ON sys.path — `python /abs/path/script.py` cannot import orn_g4 from any cwd.
PYTEST IS THE ONLY GATE — no lint, no typecheck, no CI. Run it before claiming done.
VIOLATE ANY → the failure surfaces later, in someone else's session, as a silent wrong result.
```

## What this is

26 modules under `orn_g4/` (~11.5k LOC) + `tests/` + `tools/`. A library with focused CLI utilities in `tools/` — no notebooks tracked in-tree (they live in the Azrael Colab projects). Every claim below was probed against running code and verified with the test suite.

## Ramp-up read order

Cheapest first; stop as soon as you have what you need.

1. `orn_g4/__init__.py` — module map + artifact taxonomy (A–D) + QTEA toolchain.
2. `orn_g4/config.py` — `RunConfig`, `RunDirectoryLayout`, source handles, frozen release figures.
3. `orn_g4/controller.py` — the 11-stage resumable graph.
4. `orn_g4/qtea_hf.py` / `tools/qtea_hf.py` — QTEA Hugging Face audit, lossless packing, verification, publishing, and full-model loading.
5. The module you are changing.
6. `tests/test_<module>.py` — the executable spec for that module.

## Commands

```bash
python -m pytest tests/ -q                          # 144 passed, 2 skipped; CPU-only, no network
python -m pytest tests/test_qtea_hf.py -q           # 10 QTEA HF integration tests
python -m pytest tests/test_data.py -q              # one module
python -m pytest tests/test_data.py::test_name -q   # one test
```

- Not pip-installed in a clean checkout: `tests/conftest.py` puts the repo root on `sys.path`, so pytest works from any cwd.
- No lint / typecheck / formatter config exists (no `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`). pytest is the whole gate.
- Test deps: numpy + torch; pyarrow for `FrozenDataset` and for the frozen-release preflight (`frozen_release.build_frozen_mvp_layout` reads Parquet metadata). No GPU, no HF network — HF/Drive paths are mocked with synthetic fixtures built in `tmp_path`.

## Import resolution

`import orn_g4` resolves only when the repo root is on `sys.path`. Measured matrix:

| Invocation | Result |
|---|---|
| cwd = repo root, `python -c "import orn_g4"` | OK |
| cwd = repo root, `python /abs/path/script.py` | **ModuleNotFoundError** |
| cwd = elsewhere, `python -c "import orn_g4"` | ModuleNotFoundError |
| cwd = elsewhere, `PYTHONPATH=<repo> python script.py` | OK |
| `pytest tests/` from any cwd | OK (conftest inserts repo root) |

`sys.path[0]` is the *script's* directory, not the cwd — an out-of-tree script needs `PYTHONPATH=<repo>` or `pip install -e .`.

## Name corrections

Every dangling `:func:` cross-reference in this repo lives in `orn_g4/errors.py` — 8 references across 6 modules (of 23 `:func:` references in `orn_g4/errors.py`, 33 repo-wide). Resolve against the module namespace, never the docstring:

| Cited in `orn_g4/errors.py` | Actually |
|---|---|
| `orn_g4.ternary.group_scales` | `orn_g4.ternary_core.group_scales` |
| `orn_g4.ternary.hard_ternary_forward` | `orn_g4.ternary_core.hard_ternary_forward` |
| `orn_g4.ternary.sequential_gptq` | `orn_g4.ternary_gptq.sequential_gptq` |
| `orn_g4.rotation.apply_hadamard` | `orn_g4.rotation.signed_hadamard` / `rotate_weights` |
| `orn_g4.lora.apply_lora` | `orn_g4.lora.apply_lora_to_model` |
| `orn_g4.recovery.selective_qat` | `orn_g4.recovery.selective_qat_tracking` / `train_qat` |
| `orn_g4.export.write_canonical_checkpoint` | `orn_g4.export.write_canonical_ternary_checkpoint` |
| `orn_g4.utils.resolve_package_root` | no replacement — it does not exist |

`orn_g4/errors.py` also defines `RotationError`, but `orn_g4/__init__.py` lists it in `__all__` without importing it — `from orn_g4 import *` raises `AttributeError: RotationError`.

## Module map

Non-obvious ownership only; the rest is self-describing.

| Module | Owns |
|---|---|
| `orn_g4/config.py` | `RunConfig` (immutable run contract), `RunDirectoryLayout` (on-disk tree), `ExecutionManifest` (resume state), source handles, frozen release figures |
| `orn_g4/controller.py` | the 11-stage resumable graph (`STAGE_ORDER`) and `CELL_TO_STAGE` |
| `orn_g4/checkpoint_spec.py` | the canonical on-disk ternary format + `validate_standalone_completeness` |
| `orn_g4/export.py` | artifacts A–D (PEFT adapter, canonical ternary, trainable continuation, dense/GGUF) |
| `orn_g4/loader.py` | `save_canonical`/`load_canonical`, `module_replace`, trainable checkpoints, RoPE buffer restore |
| `orn_g4/ternary_core.py` | the math, numpy-first so it is unit-testable CPU-only |
| `orn_g4/recovery.py` | scale distillation, low-rank residual, selective QAT, teacher-guided mixed-precision recovery (`train_mixed_precision_recovery`) |
| `orn_g4/qtea.py` | core QTEA mathematics, dead-zone search, iterative ternary fitting (ITF), `QTEALinear`, mixed-precision policy (`PROJECTION_PRECISION_POLICY`, `discover_mixed_precision_targets`, `convert_model_mixed_precision`) |
| `orn_g4/quant4.py` | protected 4-bit affine codec (`Protected4BitLinear`, `affine_uint4`) with optional rank-16 recovery adapters |
| `orn_g4/fallback.py` | independent Q4 fallback (`build_q4_fallback_model`, `export_q4_fallback_package`, `load_q4_fallback_package`) directly from BF16 teacher |
| `orn_g4/qtea_hf.py` | first-party Hugging Face QTEA pipeline: remote audit, lossless shard packing, verification, staging branch publish, complete-model meta-skeleton instantiation with tied weights |
| `tools/qtea_hf.py` | operational CLI for QTEA: `audit`, `pack`, `verify`, `publish`, `benchmark` |
| `orn_g4/frozen_release.py` | read-only verification of the seven frozen MVP shards against the SHA-256 sidecars that ship with the release; **no in-package caller** — the notebook drives it |
| `orn_g4/source_contract.py` | fail-closed Hub source pinning (`resolve_hub_sha`, `enforce_source_lock`), manifest-driven release validation, chat-template alignment; **no in-package caller** — the notebook drives it |
| `orn_g4/gates.py`, `orn_g4/evaluation.py`, `orn_g4/reporting.py` | verification gates A–F; offline eval; report labelled `UNVERIFIED-IN-THIS-ENVIRONMENT` |

`orn_g4/safetensors_io.py` implements the HF wire format; `orn_g4/utils.py` holds atomic writes and streaming sha256; `orn_g4/env.py`, `orn_g4/template.py`, `orn_g4/errors.py` are the env inspector, chat-template fingerprint, and error vocabulary.

## Traps

1. **Two disjoint stage-naming systems.** `config.CELL_STAGE_IDS` (14 names like `STAGE_00_ENV_RESOLVE`) and `controller.STAGE_ORDER` (11 names like `IMPORT_MODEL_AND_DATA`) share no names. `controller.CELL_TO_STAGE` maps cell index → `STAGE_ORDER` name. Do not try to reconcile them.
2. **Three canonical-checkpoint writers.** `checkpoint_spec.write_canonical_ternary_checkpoint` is the real one; `loader.save_canonical`/`load_canonical` saves a whole `nn.Module` as sharded safetensors (refuses an existing dir → `FileExistsError`; stages into `<name>.incomplete` first); `export.write_canonical_ternary_checkpoint` is a backward-compat wrapper delegating to `export.write_canonical_ternary_artifact`. Gates and `export.load_effective_ternary` read via `checkpoint_spec.load_canonical_ternary_checkpoint`.
3. **Version drift:** `pyproject.toml` = `0.1.0`, `orn_g4.__version__` = `2.0.0`, `RunConfig.schema_version` = `2.0.0` (the last is what lands in manifests).
4. **`config.Settings` is dead code** — nothing imports it. Its `MODE` (STATIC/SMOKE/FULL) knob is not wired to anything, and `orn_g4/__init__.py`'s claim of an `ORN_G4_GATE` env var is stale: the only env var read anywhere is `PRISM_HOME` (`orn_g4/gates.py`).
5. **`TernaryLinear` is `None` without torch** (`orn_g4/ternary_core.py` sets it to `None` in a module-level `except ImportError`), so `TernaryLinear.from_linear(...)` fails with `AttributeError`, not `ImportError`. `loader.save_canonical` guards with `if TernaryLinear is not None` for exactly this reason.
6. **Two frozen-release contracts, deliberately in tension.** `frozen_release.inspect_frozen_mvp` hard-asserts the legacy MVP figures (exactly 7 shards, exactly 17,848 records), while `source_contract.validate_release` treats those counts as *non*-authoritative and validates against the release manifest instead. `orn_g4/config.py` now labels `EXPECTED_DATASET_RECORDS` / `EXPECTED_DATASET_SHARDS` / `EXPECTED_TEMPLATE_SHA256_PREFIX` / `EXPECTED_TEMPLATE_REVISION` as `LEGACY MVP ONLY`. Pick the contract that matches the release you are running; do not "reconcile" them by loosening the MVP assertions.
7. **R-26 rule** (`export._assert_not_source_fp`): a checkpoint whose tensors are all floating-point (no int8 codes) under a ternary filename raises `ExportError`.
8. **Undeclared runtime deps.** `kagglehub` and `peft` are imported by the code but absent from `pyproject.toml` dependencies — `pip install -e .` will not bring them in. `peft` is only needed by `lora.setup_peft_lora`. (`huggingface_hub` *is* declared, as `>=0.28.0`.)
9. **`transformers` may not import.** Probe `python -c "import transformers"` before relying on `modeling.discover_architecture`, `load_foundation_model`, `load_processor_and_tokenizer`, or `orn_g4/template.py`. At `61ab3ac` it still fails: `huggingface-hub>=1.5.0,<2.0` required, found `2.4.0`. numpy/torch paths are unaffected. The declared floor is `transformers>=5.8.1` and the installed version is `5.16.1`, so the floor is satisfied — the break is the *upper* bound on `huggingface_hub`, and because `pyproject.toml` declares `huggingface_hub>=0.28.0` with no ceiling, `pip install -e .` will **not** repair it. Fixing it means pinning `huggingface_hub<2.0` in the environment, not editing the floor.
10. **The two preflight modules are notebook-facing, not wired in.** `frozen_release` and `source_contract` have no caller anywhere inside `orn_g4/` — only their own tests import them. They are consumed by the Azrael-g4 notebook (PR #1), which is not tracked in this repo, so "the notebook calls it" is an inference from the PR description and not a fact you can verify here. The same is true of the new `revision` / `allow_drive_fallback` arguments: `controller.py` never calls `build_layout`, `locate_dataset`, `import_model`, or `load_foundation_model`.
11. **QTEA dynamic buffer shapes:** `QTEALinear` registers placeholder `(0,)` / `(0, 0)` buffers for `salient_cols`, `residual_index`, `residual_value`, and `residual_scale` because the salient column count `k` varies per projection. `QTEALinear._load_from_state_dict` resizes those four dynamic buffers before delegating to `nn.Module._load_from_state_dict`, while `load_qtea_full_model` strictly enforces fixed-geometry shapes (`ternary`, `col_v`, `group_alpha`, `group_beta`, `bias`) and cross-checks dynamic residual shapes against `salient_cols.numel()`.
12. **Stale BF16 asset hazard:** Original uncompressed model assets include `model.safetensors.index.json` and `model-*.safetensors`. In the sharded QTEA distribution (`qtea-sharded-v2`), these stale files are purged from `assets/` to prevent Hugging Face loaders from attempting to resolve dense BF16 shards.
13. **Main branch protection:** `moeshawky/Azrael-QTEA-9B@main` is immutable and quality-locked. Programmatic publish tools refuse to target `main` or the source revision. Target branch is `qtea-sharded-v2`.

## QTEA Hugging Face Toolchain & Pipeline

Maintained first-party pipeline (`orn_g4/qtea_hf.py` and `tools/qtea_hf.py`) replacing fragmented prototype scripts and legacy Google Drive dependencies.

### Hardware Requirements
- **Packing, Audit, Verification**: CPU-sufficient. Operates cleanly with 2–4 GiB system RAM and ~15–20 GiB temporary disk space for local shard staging.
- **Full-Model Inference**: Requires NVIDIA CUDA GPU with $\ge 24\text{ GiB}$ VRAM (e.g. RTX 3090/4090, A5000, A100). The current reference forward pass materializes effective dense weights during execution; no packed-kernel speedup is claimed. Model retains provenance status `QUALITY_REJECTED_RECOVERY_REQUIRED` (+1.8708 CE degradation over teacher).

### Operational CLI Commands

```bash
# 1. Remote read-only audit (inspects inventory, validates 248 projections, calculates download size without fetching weights)
python tools/qtea_hf.py audit --repo moeshawky/Azrael-QTEA-9B

# 2. Local lossless repacking into SafeTensors shards (preserves FP8 residuals and int dtypes bit-for-bit)
python tools/qtea_hf.py pack --source /path/to/downloaded_qtea --output /path/to/sharded_out --shard-mib 2048

# 3. Independent cryptographic verification of packed package
python tools/qtea_hf.py verify --path /path/to/sharded_out

# 4. Direct HF download, local repack, and upload to staging branch (fails closed if token is invalid or owner mismatches)
python tools/qtea_hf.py publish --repo moeshawky/Azrael-QTEA-9B --branch qtea-sharded-v2 --workdir /tmp/qtea-work --token "$HF_TOKEN"

# 5. Full-model reconstruction on meta device and GPU inference benchmark (records separated loading and inference peak VRAM)
python tools/qtea_hf.py benchmark --model-path /path/to/sharded_out --device cuda --prompt "Explain the difference between synchronous and asynchronous consensus."
```

## Data contract (`orn_g4/data.py`)

- `FrozenDataset` requires Parquet shards with columns `id, source, token_count, supervised_count` (index/metadata) and `input_ids, attention_mask, labels` (payload).
- Per row it enforces: `len(mask) == len(labels) == len(ids)`; declared `token_count`/`supervised_count` match the arrays; `1 <= n <= 8192`; mask ∈ {0,1}; **right-padding only** (any other pattern → `DataAlignmentError: non-right padding requires dedicated collator`); padded positions must not be supervised; labels must be **unshifted** and position-aligned with `input_ids` at supervised positions; token ids inside the tokenizer vocab (only checked when a tokenizer is passed to the constructor).
- `eval_firewall/` is **always** excluded from training and calibration (`resolve_shards` hardcodes it).
- `resolve_shards` reads the shard list from the manifest under `shards` / `train_shards` / `files` / `training_shards` (list or dict-of-lists) and raises `DatasetError` if a declared shard is missing. It never silently guesses the layout.
- Frozen release figures (`orn_g4/config.py`, marked `LEGACY MVP ONLY` — the frozen *MVP subset* figures, not archive-wide totals): 7 shards, 17,848 records, 30,029,360 total tokens, 8,748,755 supervised tokens, max_seq_len 8192, EOS id 248046, template sha256 prefix `9dd2fbd2`, revision `33e8951`. The release manifest declares 17,999 archive-wide records — 151 more than the MVP subset, held in a separate persona shard. Never compare archive-wide totals against the narrower `train_ready/mvp/` subset.

## Source resolution

- `modeling.import_model(handle)`: local dir containing the model's own `config.json` → HF snapshot (`hf://owner/repo` or `owner/repo`) → KaggleHub (`kaggle://…` or any handle with more than one `/`) → else `ModelSourceError`.
- `data.locate_dataset(source)`: local path → HF dataset snapshot (`hf://`, the exact `HF_DATASET_HANDLE`, or *any* string containing `/`) → Drive ZIP **only when `allow_drive_fallback=True`** (default `False` — the Drive tier is opt-in and the resolver fails closed by default). The `owner/repo` heuristic means a non-existent relative path containing `/` is treated as an HF repo id. `revision=None` pins the Hub snapshot to an immutable commit when set.
- Handles live in `orn_g4/config.py`, the single source of truth: `HF_MODEL_HANDLE = moeshawky/Ornith-1.5-9B`, `HF_DATASET_HANDLE = moeshawky/azrael-dataset`, `KAGGLE_MODEL_HANDLE`, `TEMPLATE_REPO`, `DRIVE_DATASET_ZIP`, `DRIVE_RUN_ROOT`.

## Recovery invariants (`orn_g4/recovery.py`)

- `train_recovery(model, ds, stage, budget, lr, run_dir, checkpoint_every)`; `stage ∈ {"scale", "residual"}`. `budget` is a **supervised-token** budget (`progress["tokens"] += supervised`), not steps.
- Scales are clamped to ≥ 1e-8 after every optimizer step; non-finite loss or gradients raise `RecoveryError`; grad-norm is clipped to 1.0.
- **Lineage is enforced:** resuming from a checkpoint whose recorded `stage` differs, or whose `canonical_manifest_sha256` differs from the current manifest, raises `RecoveryError`.
- `train_qat` only trains modules that are `TernaryLinear` **and** have `shadow is not None` (built with `qat=True`); everything else is frozen and silently skipped.

## Conventions to preserve

- Public API docstrings carry an explicit `# Errors:` note naming the exception type and the operator remediation (63 such notes in the package). This is the repo's contract style — keep it.
- All exceptions derive from `OrnG4Error`; do not leak raw third-party exceptions across module boundaries. Every error *class* is defined in `orn_g4/errors.py` — feature modules import it from there. A feature module that defines its own exception type breaks the single catch-all: `except OrnG4Error` will not catch it.
- Writes go through `utils.write_json_atomic` / `utils.atomic_write_bytes` (temp file + rename); checksums via `utils.sha256_file`.
- `orn_g4/ternary_core.py` is numpy-first so the math stays unit-testable without torch; `orn_g4/ternary_ste.py` is the one deliberate exception with a top-level `import torch`.

## BANNED

- Adding a top-level `import torch` / `import transformers` to a module — the package must import on a CPU-only host.
- Trusting a symbol because a docstring names it — 8 of the 23 `:func:` references in `orn_g4/errors.py` do not exist.
- Writing into a saved checkpoint or artifact directory — `save_canonical` refuses an existing dir, and the controller re-runs a stage whose checksum moved.
- Importing `orn_g4.config.Settings` or switching on `MODE` — dead code; nothing reads it.
- Treating `pyproject.toml`'s `0.1.0` as the package version — `orn_g4.__version__` is `2.0.0`.
- Assuming `pip install -e .` supplies `kagglehub` or `peft` — imported but undeclared. (`huggingface_hub` is declared as `>=0.28.0`, but that floor does not satisfy `transformers`' own `huggingface-hub<2.0` requirement.)
- Adding a lint/typecheck/CI config "to match other repos" — there is none here; pytest is the gate.
- Training or calibrating on `eval_firewall/` — `resolve_shards` always excludes it.
- Publishing a file path that does not exist in the repo — a broken reference fails silently.
- Publishing to or modifying Hugging Face `main` branch on `moeshawky/Azrael-QTEA-9B` — `main` is protected; target `qtea-sharded-v2`.
- Passing QTEA checkpoints through pure-ternary serializers or NumPy float conversions that destroy FP8 residuals.
- Claiming packed-kernel acceleration or production-ready status for QTEA inference while evaluated at `QUALITY_REJECTED_RECOVERY_REQUIRED`.

---

*The docstrings are not the API. Probe the namespace before you trust a name.*
