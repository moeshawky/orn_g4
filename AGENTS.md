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

23 modules under `orn_g4/` (~9.8k LOC) + `tests/`. A library only — no CLI, no `__main__`, no notebooks: 22 lines in the package reference `cell_00..cell_13`, but 0 files named `cell_*` are tracked here (they live in the Azrael-g4 Colab project). Docstrings cite an external "work-order §N" spec that is not in this repo — the code is the only source of truth for what was built. Every claim below was probed at commit `3af2962`.

## Ramp-up read order

Cheapest first; stop as soon as you have what you need.

1. `orn_g4/__init__.py` — module map + artifact taxonomy (A–D).
2. `orn_g4/config.py` — `RunConfig`, `RunDirectoryLayout`, source handles, frozen release figures.
3. `orn_g4/controller.py` — the 11-stage resumable graph.
4. The module you are changing.
5. `tests/test_<module>.py` — the executable spec for that module.

## Commands

```bash
python -m pytest tests/ -q                          # 63 passed at 3af2962, ~5s warm, CPU-only, no network
python -m pytest tests/test_data.py -q              # one module
python -m pytest tests/test_data.py::test_name -q   # one test
```

- Not pip-installed in a clean checkout: `tests/conftest.py` puts the repo root on `sys.path`, so pytest works from any cwd.
- No lint / typecheck / formatter config exists (no `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`). pytest is the whole gate.
- Test deps: numpy + torch; pyarrow only for `FrozenDataset`. No GPU, no HF network — HF/Drive paths are mocked with synthetic fixtures built in `tmp_path`.

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

Every dangling `:func:` cross-reference in this repo lives in `orn_g4/errors.py` — 8 references across 6 modules (of 19 `:func:` references repo-wide). Resolve against the module namespace, never the docstring:

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
| `orn_g4/recovery.py` | scale distillation, low-rank residual, selective QAT |
| `orn_g4/gates.py`, `orn_g4/evaluation.py`, `orn_g4/reporting.py` | verification gates A–F; offline eval; report labelled `UNVERIFIED-IN-THIS-ENVIRONMENT` |

`orn_g4/safetensors_io.py` implements the HF wire format; `orn_g4/utils.py` holds atomic writes and streaming sha256; `orn_g4/env.py`, `orn_g4/template.py`, `orn_g4/errors.py` are the env inspector, chat-template fingerprint, and error vocabulary.

## Traps

1. **Two disjoint stage-naming systems.** `config.CELL_STAGE_IDS` (14 names like `STAGE_00_ENV_RESOLVE`) and `controller.STAGE_ORDER` (11 names like `IMPORT_MODEL_AND_DATA`) share no names. `controller.CELL_TO_STAGE` maps cell index → `STAGE_ORDER` name. Do not try to reconcile them.
2. **Three canonical-checkpoint writers.** `checkpoint_spec.write_canonical_ternary_checkpoint` is the real one; `loader.save_canonical`/`load_canonical` saves a whole `nn.Module` as sharded safetensors (refuses an existing dir → `FileExistsError`; stages into `<name>.incomplete` first); `export.write_canonical_ternary_checkpoint` is a backward-compat wrapper delegating to `export.write_canonical_ternary_artifact`. Gates and `export.load_effective_ternary` read via `checkpoint_spec.load_canonical_ternary_checkpoint`.
3. **Version drift:** `pyproject.toml` = `0.1.0`, `orn_g4.__version__` = `2.0.0`, `RunConfig.schema_version` = `2.0.0` (the last is what lands in manifests).
4. **`config.Settings` is dead code** — nothing imports it. Its `MODE` (STATIC/SMOKE/FULL) knob is not wired to anything, and `orn_g4/__init__.py`'s claim of an `ORN_G4_GATE` env var is stale: the only env var read anywhere is `PRISM_HOME` (`orn_g4/gates.py`).
5. **`TernaryLinear` is `None` without torch** (`orn_g4/ternary_core.py` sets it to `None` in a module-level `except ImportError`), so `TernaryLinear.from_linear(...)` fails with `AttributeError`, not `ImportError`. `loader.save_canonical` guards with `if TernaryLinear is not None` for exactly this reason.
6. **`README.md`'s "Package Structure" is stale** — it omits `orn_g4/checkpoint_spec.py`, `orn_g4/controller.py`, `orn_g4/evaluation.py`, `orn_g4/reporting.py`, and claims "60 passed" (63 at `3af2962`).
7. **R-26 rule** (`export._assert_not_source_fp`): a checkpoint whose tensors are all floating-point (no int8 codes) under a ternary filename raises `ExportError`.
8. **Undeclared runtime deps.** `huggingface_hub`, `kagglehub`, `peft` are imported by the code but absent from `pyproject.toml` dependencies — `pip install -e .` will not bring them in. `peft` is only needed by `lora.setup_peft_lora`.
9. **`transformers` may not import.** Probe `python -c "import transformers"` before relying on `modeling.discover_architecture`, `load_foundation_model`, `load_processor_and_tokenizer`, or `orn_g4/template.py`. At `3af2962` it fails: `huggingface-hub>=1.5.0,<2.0` required, found `2.4.0`. numpy/torch paths are unaffected.
10. **Unmerged branch.** `origin/fix/g4-source-run-contract-20261109` is not on `main` and adds two modules absent from the working tree — do not import them.

## Data contract (`orn_g4/data.py`)

- `FrozenDataset` requires Parquet shards with columns `id, source, token_count, supervised_count` (index/metadata) and `input_ids, attention_mask, labels` (payload).
- Per row it enforces: `len(mask) == len(labels) == len(ids)`; declared `token_count`/`supervised_count` match the arrays; `1 <= n <= 8192`; mask ∈ {0,1}; **right-padding only** (any other pattern → `DataAlignmentError: non-right padding requires dedicated collator`); padded positions must not be supervised; labels must be **unshifted** and position-aligned with `input_ids` at supervised positions; token ids inside the tokenizer vocab (only checked when a tokenizer is passed to the constructor).
- `eval_firewall/` is **always** excluded from training and calibration (`resolve_shards` hardcodes it).
- `resolve_shards` reads the shard list from the manifest under `shards` / `train_shards` / `files` / `training_shards` (list or dict-of-lists) and raises `DatasetError` if a declared shard is missing. It never silently guesses the layout.
- Frozen release figures (`orn_g4/config.py`): 7 shards, 17,848 records, 30,029,360 total tokens, 8,748,755 supervised tokens, max_seq_len 8192, EOS id 248046, template sha256 prefix `9dd2fbd2`, revision `33e8951`.

## Source resolution

- `modeling.import_model(handle)`: local dir containing the model's own `config.json` → HF snapshot (`hf://owner/repo` or `owner/repo`) → KaggleHub (`kaggle://…` or any handle with more than one `/`) → else `ModelSourceError`.
- `data.locate_dataset(source)`: local path → HF dataset snapshot (`hf://`, the exact `HF_DATASET_HANDLE`, or *any* string containing `/`) → Drive ZIP. The `owner/repo` heuristic means a non-existent relative path containing `/` is treated as an HF repo id.
- Handles live in `orn_g4/config.py`, the single source of truth: `HF_MODEL_HANDLE = moeshawky/Ornith-1.5-9B`, `HF_DATASET_HANDLE = moeshawky/azrael-dataset`, `KAGGLE_MODEL_HANDLE`, `TEMPLATE_REPO`, `DRIVE_DATASET_ZIP`, `DRIVE_RUN_ROOT`.

## Recovery invariants (`orn_g4/recovery.py`)

- `train_recovery(model, ds, stage, budget, lr, run_dir, checkpoint_every)`; `stage ∈ {"scale", "residual"}`. `budget` is a **supervised-token** budget (`progress["tokens"] += supervised`), not steps.
- Scales are clamped to ≥ 1e-8 after every optimizer step; non-finite loss or gradients raise `RecoveryError`; grad-norm is clipped to 1.0.
- **Lineage is enforced:** resuming from a checkpoint whose recorded `stage` differs, or whose `canonical_manifest_sha256` differs from the current manifest, raises `RecoveryError`.
- `train_qat` only trains modules that are `TernaryLinear` **and** have `shadow is not None` (built with `qat=True`); everything else is frozen and silently skipped.

## Conventions to preserve

- Public API docstrings carry an explicit `# Errors:` note naming the exception type and the operator remediation (54 such notes in the package). This is the repo's contract style — keep it.
- All exceptions derive from `OrnG4Error`; do not leak raw third-party exceptions across module boundaries.
- Writes go through `utils.write_json_atomic` / `utils.atomic_write_bytes` (temp file + rename); checksums via `utils.sha256_file`.
- `orn_g4/ternary_core.py` is numpy-first so the math stays unit-testable without torch; `orn_g4/ternary_ste.py` is the one deliberate exception with a top-level `import torch`.

## BANNED

- Adding a top-level `import torch` / `import transformers` to a module — the package must import on a CPU-only host.
- Trusting a symbol because a docstring names it — 8 of the 19 `:func:` references in `orn_g4/errors.py` do not exist.
- Writing into a saved checkpoint or artifact directory — `save_canonical` refuses an existing dir, and the controller re-runs a stage whose checksum moved.
- Importing `orn_g4.config.Settings` or switching on `MODE` — dead code; nothing reads it.
- Treating `pyproject.toml`'s `0.1.0` as the package version — `orn_g4.__version__` is `2.0.0`.
- Assuming `pip install -e .` supplies `huggingface_hub`, `kagglehub`, or `peft` — imported but undeclared.
- Adding a lint/typecheck/CI config "to match other repos" — there is none here; pytest is the gate.
- Training or calibrating on `eval_firewall/` — `resolve_shards` always excludes it.
- Publishing a file path that does not exist in the repo — a broken reference fails silently.

---

*The docstrings are not the API. Probe the namespace before you trust a name.*
