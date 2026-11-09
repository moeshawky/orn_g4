"""orn_g4 — Ornith-1.5-9b Colab G4 SFT, ternary conversion, and recovery.

Package implementing the work-order campaign:
  * modeling      — KaggleHub import + architecture discovery (§1)
  * template      — native chat-template inspection (§2)
  * data          — Google Drive nested-ZIP dataset import (§3)
  * env           — compute-environment inspection (§4)
  * lora          — short BF16 behavioral SFT + W_adapted parity (§5)
  * ternary_core  — grouped ternary representation & primitives (§6)
  * ternary_ste   — activation-aware STE control (§6)
  * ternary_gptq  — sequential ternary GPTQ primary candidate (§6)
  * rotation      — signed Hadamard rotation experiment (§6)
  * pre_conversion_surgery_probe — P1/P2/P4 bounded surgery probes (§6)
  * recovery      — scale distillation / low-rank residual / selective QAT (§7)
  * safetensors_io — spec-compliant safetensors reader/writer (Artifacts A–D)
  * checkpoint_spec — versioned canonical ternary checkpoint spec (Artifact B)
  * loader        — portable PyTorch model loader (§3)
  * export        — Artifacts A–D + fresh-process reload + dense export (§10)
  * gates         — Verification Gates A–F (§9 / §11)
  * evaluation    — offline operational evaluation (§9)
  * reporting     — experiment report (§10-E)
  * controller    — 11-stage deterministic resumable controller (§8/§11)

Correctness-critical mathematics live here so they can be unit-tested on any
host (including GPU-less staging hosts). torch/transformers/kagglehub/safetensors
are imported lazily inside functions where needed.

Artifact architecture (work-order §2):
  A. BF16 adaptation artifact — PEFT-compatible adapter_model.safetensors
  B. Canonical effective ternary checkpoint — sharded safetensors, standalone
  C. Trainable continuation state — exact_resume / portable_continuation
  D. Runtime deployment — PTQ1_0/GGUF candidate from final recovered canonical

Execution gates (env ORN_G4_GATE): STATIC_CHECK / SMOKE_TEST / FULL_RUN.
GPU-bound stages are labelled UNVERIFIED-IN-THIS-ENVIRONMENT on this host.
"""

from __future__ import annotations

from .config import (
    CELL_STAGE_IDS,
    DRIVE_DATASET_ZIP,
    DRIVE_RUN_ROOT,
    EXPECTED_DATASET_RECORDS,
    EXPECTED_DATASET_SHARDS,
    EXPECTED_MAX_SEQ_LEN,
    EXPECTED_SUPERVISED_TOKENS,
    EXPECTED_TOTAL_TOKENS,
    DEFAULT_MODEL_HANDLE,
    HF_MODEL_HANDLE,
    KAGGLE_MODEL_HANDLE,
    TEMPLATE_REPO,
    ExecutionManifest,
    RunConfig,
    RunDirectoryLayout,
    StageResult,
    TernaryArtifact,
)
from .controller import (
    CELL_TO_STAGE,
    STAGE_DONE,
    STAGE_ORDER,
    ExecutionController,
)
from .errors import (
    BudgetExceededError,
    ChecksumMismatchError,
    ControllerError,
    DataAlignmentError,
    DatasetError,
    ExportError,
    LoRAError,
    ModelSourceError,
    OrnG4Error,
    PackageResolutionError,
    RecoveryError,
    TemplateError,
    TernaryError,
)

__version__ = "2.0.0"

__all__ = [
    # config
    "DEFAULT_MODEL_HANDLE",
    "HF_MODEL_HANDLE",
    "KAGGLE_MODEL_HANDLE",
    "TEMPLATE_REPO",
    "DRIVE_DATASET_ZIP",
    "DRIVE_RUN_ROOT",
    "EXPECTED_DATASET_RECORDS",
    "EXPECTED_DATASET_SHARDS",
    "EXPECTED_TOTAL_TOKENS",
    "EXPECTED_SUPERVISED_TOKENS",
    "EXPECTED_MAX_SEQ_LEN",
    "CELL_STAGE_IDS",
    "RunConfig",
    "RunDirectoryLayout",
    "StageResult",
    "TernaryArtifact",
    "ExecutionManifest",
    # errors
    "OrnG4Error",
    "PackageResolutionError",
    "ModelSourceError",
    "TemplateError",
    "DatasetError",
    "DataAlignmentError",
    "LoRAError",
    "TernaryError",
    "RotationError",
    "RecoveryError",
    "ExportError",
    "ControllerError",
    "BudgetExceededError",
    "ChecksumMismatchError",
    # controller
    "ExecutionController",
    "STAGE_ORDER",
    "STAGE_DONE",
    "CELL_TO_STAGE",
]
