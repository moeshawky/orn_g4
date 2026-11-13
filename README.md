# `orn-g4`

> **Deterministic 1.58-bit ternary quantization, low-rank recovery, and canonical safetensors serialization for Ornith-1.5-9B and Qwen3.5 architectures.**

[![Tests](https://img.shields.io/badge/tests-150%20passed%2C%202%20skipped-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)]()
[![Framework](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)]()
[![HuggingFace](https://img.shields.io/badge/Transformers-5.8.1%2B-yellow.svg)]()

`orn_g4` is a high-precision quantization, recovery, and deployment engine designed for compressing 9.65-billion-parameter hybrid models (transformer blocks, Gated DeltaNet recurrent states, vision towers, and multimodal projection layers) into ternary representation ($\{-1, 0, +1\}$ discrete codes with FP32 group scales) while maintaining behavioral parity and numerical stability under tight VRAM constraints (15–24 GiB).

---

## Key Highlights

- **Native Multimodal Support**: Automatic architecture discovery and layer resolution for `Qwen3_5ForConditionalGeneration` and multimodal causal LM configurations.
- **Dual-Quantizer Tournament**: Evaluates each projection against held-out activations, dynamically selecting the winner between activation-diagonal ternarization (`ternarize_diag`) and group-local second-order error feedback (`ternarize_gptq_group`).
- **Memory-Bounded Forward Hooks**: Collects activation covariance matrices without VRAM explosions via linspace downsampling (256 tokens/module ceiling).
- **Two-Stage Decoupled Recovery**: Optimizes group scales ($\text{scales} \ge 10^{-8}$) before residual adapters, preventing gradient explosion.
- **Selective STE QAT**: Parameter-bounded Straight-Through Estimator fine-tuning with FP32 shadows and audited discrete code flips.
- **Canonical Safetensors Serialization**: 512MB chunked safetensors export with Hugging Face specification compliance (`__metadata__` wire format) and automatic non-persistent RoPE buffer (`inv_freq`) reconstruction upon reloading.
- **Parquet Dataset Hygiene**: Preflight-verified Parquet shard loading with bitwise duplicate-closure clustering, right-padding assertions, and unshifted position-aligned label validation.
- **Fail-Closed Source Provenance**: Model and dataset inputs pinned to immutable Hugging Face commit SHAs, with the frozen seven-shard MVP training subset verified read-only against the SHA-256 proofs that ship with the release — no dataset file is created, modified, or regenerated.

---

## Mathematical Formulation

### 1. Grouped Ternary Representation

Each linear weight matrix $W \in \mathbb{R}^{O \times I}$ is quantized in groups of $G = 128$ consecutive input channels:

$$W_{\text{effective}} = \text{dequantize}(C, S) + B A$$

where:
- $C \in \{-1, 0, +1\}^{O \times \lceil I / G \rceil \times G}$ is stored as signed 8-bit integers (`int8`).
- $S \in \mathbb{R}^{O \times \lceil I / G \rceil}$ stores per-group floating-point scaling factors (`float32`).
- $B \in \mathbb{R}^{O \times r}$ and $A \in \mathbb{R}^{r \times I}$ are optional FP32 low-rank residual correction matrices ($r \in \{16, 32\}$).

### 2. Group-Local Dead-Zone Quantization

For weight row segment $w \in \mathbb{R}^G$:
$$s = \frac{1}{|w|_{>0}} \sum_{j=1}^G |w_j|$$
$$c_j = \begin{cases} +1, & w_j > 0.5 s \\ -1, & w_j < -0.5 s \\ 0, & |w_j| \le 0.5 s \end{cases}$$

---

## Installation

### From GitHub

```bash
pip install -q git+https://github.com/moeshawky/orn_g4.git
```

### Local Editable Install

```bash
git clone https://github.com/moeshawky/orn_g4.git
cd orn_g4
pip install -e .
```

---

## Quickstart

### 1. Load Foundation Model

```python
import torch
from orn_g4.modeling import import_model, load_foundation_model, load_processor_and_tokenizer

# Primary Hugging Face model repository (or Kaggle handle / local path)
model_handle = "moeshawky/Ornith-1.5-9B"

# Automatically downloads from Hugging Face without requiring Kaggle API tokens
model_dir = import_model(model_handle)
model = load_foundation_model(model_dir, device="cuda", torch_dtype=torch.bfloat16)
processor, tokenizer = load_processor_and_tokenizer(model_dir)
```

### 2. Import Frozen Dataset

```python
from orn_g4.data import import_dataset, build_layout

# Primary Hugging Face dataset repository (or a local directory)
dataset_source = "hf://moeshawky/azrael-dataset"

# Downloads snapshot from Hugging Face and verifies release manifest.
# The Google Drive tier is opt-in: pass allow_drive_fallback=True to enable it.
# Pin an immutable commit with revision="<sha>" when the run must be reproducible.
layout = build_layout(source=dataset_source)
print(f"Dataset root: {layout.dataset_root}")
print(f"Resolved shards: {len(layout.shards)}")
```

### 3. Convert Linear Layers to Ternary

```python
import torch
from orn_g4.ternary_core import TernaryLinear, ternarize_diag
from orn_g4.loader import module_replace

# Quantize a layer
linear = model.model.layers[0].self_attn.q_proj
codes, scales = ternarize_diag(linear.weight.data, group_size=128)

# Replace with TernaryLinear module
ternary_layer = TernaryLinear.from_linear(linear, codes, scales, rank=16)
module_replace(model, "model.layers.0.self_attn.q_proj", ternary_layer)
```

### 4. Save & Reload Canonical Checkpoints

```python
from pathlib import Path
from orn_g4.loader import save_canonical, load_canonical

out_dir = Path("./canonical_ternary_checkpoint")

# Save 512MB-chunked safetensors with manifest and SHA-256 commitments
save_canonical(model, out_dir, source_id="ornith-1.5-9b")

# Reload in a clean process with automatic RoPE inv_freq reconstruction
reloaded_model, manifest = load_canonical(out_dir, device="cuda")
```

### 5. Two-Stage Decoupled Recovery

```python
from orn_g4.recovery import train_recovery

# Stage 1: Scale-only distillation with scale clamping (scales >= 1e-8)
train_recovery(
    model=model,
    ds=dataset,
    stage="scale",
    budget=1_000_000,
    lr=1e-3,
    run_dir="./recovery_run",
    checkpoint_every=50,
)

# Stage 2: Low-rank residual distillation
train_recovery(
    model=model,
    ds=dataset,
    stage="residual",
    budget=2_000_000,
    lr=1e-4,
    run_dir="./recovery_run",
    checkpoint_every=50,
)
```

---

## QTEA Hugging Face Toolchain

First-party tooling for remote auditing, lossless repacking, cryptographic verification, staging branch publishing, and full-model evaluation of QTEA (Quantized Ternary with Error-Adaptive residuals) models (`moeshawky/Azrael-QTEA-9B`).

### Hardware Requirements
- **Packing, Verification, Audit**: **CPU-sufficient**. Requires only ~2–4 GiB system RAM and temporary disk space for local shard staging. No GPU required.
- **Full-Model Inference**: Requires an NVIDIA CUDA GPU with $\ge 24\text{ GiB}$ VRAM (e.g. RTX 3090/4090, A5000, A100). The current reference forward path materializes effective dense weights during execution; no packed-kernel speedup is claimed. Model retains provenance status `QUALITY_REJECTED_RECOVERY_REQUIRED` (+1.8708 CE degradation over teacher).

### CLI Workflow

```bash
# 1. Audit remote Hugging Face repository (248 projections, download payload accounting)
python tools/qtea_hf.py audit --repo moeshawky/Azrael-QTEA-9B

# 2. Lossless repacking into indexed SafeTensors shards (preserves FP8 residuals and int dtypes bit-for-bit)
python tools/qtea_hf.py pack --source /path/to/downloaded_qtea --output /path/to/sharded_out --shard-mib 2048

# 3. Independent cryptographic verification of shards and manifest
python tools/qtea_hf.py verify --path /path/to/sharded_out

# 4. Direct download, local repack, and upload to staging branch (main branch is protected)
python tools/qtea_hf.py publish --repo moeshawky/Azrael-QTEA-9B --branch qtea-sharded-v2 --workdir /tmp/qtea-work --token "$HF_TOKEN"

# 5. Full-model load on meta skeleton and GPU inference benchmark
python tools/qtea_hf.py benchmark --model-path /path/to/sharded_out --device cuda --prompt "Explain the difference between synchronous and asynchronous consensus."
```

### Python API

All CLI operations are backed by importable Python functions under `orn_g4.qtea_hf`:

```python
from orn_g4.qtea_hf import (
    audit_remote_qtea,
    pack_qtea_shards,
    verify_qtea_shards,
    publish_qtea_sharded,
    load_qtea_full_model,
    benchmark_qtea_inference,
)

# Load full model without downloading 18GB BF16 source weights
model, tokenizer, meta = load_qtea_full_model("/path/to/sharded_out", device="cuda")

# Benchmark inference with separated loading and generation peak VRAM accounting
report = benchmark_qtea_inference(
    model=model,
    tokenizer=tokenizer,
    prompt="Explain quantum annealing.",
    device="cuda",
    max_new_tokens=32,
)
print(f"Generated text: {report['output_text']}")
print(f"Peak VRAM during inference: {report['inference_peak_gib']} GiB")
```

---

## Package Structure

```
orn_g4/
├── __init__.py                  # Public exports & versioning
├── checkpoint_spec.py           # Canonical on-disk ternary format & standalone completeness
├── config.py                    # RunConfig, directory layout, source handles, release figures
├── controller.py                # 11-stage deterministic resumable controller
├── data.py                      # FrozenDataset, Parquet streaming, bitwise duplicate closure
├── env.py                       # Hardware inspection (GPU, VRAM, BF16, CUDA compute)
├── errors.py                    # Explicit domain exceptions (Alignment, Gate, Recovery, QTEA)
├── evaluation.py                # Offline operational evaluation
├── export.py                    # Artifact packaging, dense BF16 materialization
├── frozen_release.py            # Read-only frozen MVP shard & checksum verification
├── gates.py                     # Execution gates & verification assertions
├── loader.py                    # Canonical checkpoint persistence & RoPE restoration
├── lora.py                      # PEFT LoRA wiring, input require grads, target filtering
├── modeling.py                  # KaggleHub import & multimodal architecture discovery
├── pre_conversion_surgery_probe.py # Bounded activation collection & P1/P2/P4 surgical probes
├── qtea.py                      # QTEA core mathematics, ITF fitting, and QTEALinear module
├── qtea_hf.py                   # Hugging Face QTEA pipeline: audit, lossless pack, verify, publish, full-model loading
├── recovery.py                  # Two-stage recovery, scale clamping, and selective QAT
├── reporting.py                 # Experiment report generation
├── rotation.py                  # Signed Hadamard rotation transforms
├── safetensors_io.py            # Spec-compliant sharded safetensors reader/writer
├── source_contract.py           # Fail-closed Hub source pinning & release validation
├── template.py                  # Native chat-template inspection & SHA256 verification
├── ternary_core.py              # GroupedTernary, TernaryLinear, pack/unpack kernels
├── ternary_gptq.py              # Second-order group-local GPTQ error feedback
├── ternary_ste.py               # Straight-Through Estimator forward controls
└── utils.py                     # Atomic JSON, SHA-256 streaming, memory tracking

tools/
└── qtea_hf.py                   # Operational CLI: audit, pack, verify, publish, benchmark
```

---

## Verification & Testing

The package includes a comprehensive unit test suite covering quantization math, numerical stability, streaming Parquet data hygiene, safetensors serialization, and the complete QTEA Hugging Face pipeline.

Run the test suite:

```bash
python -m pytest tests/ -q
```

```
...............................................................ss..............................................................................................
============================== 144 passed, 2 skipped in 231.85s ==============================
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
