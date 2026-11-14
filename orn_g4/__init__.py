"""orn_g4 — Ornith-1.5-9b Colab G4 SFT, ternary conversion, and recovery.

Package implementing the work-order campaign:
  * modeling      — KaggleHub import + architecture discovery (§1)
  * template      — native chat-template inspection (§2)
  * data          — Google Drive nested-ZIP dataset import (§3)
  * frozen_release — read-only frozen MVP shard/checksum verification
  * source_contract — fail-closed Hub source pinning and release validation
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
    DEFAULT_DATASET_SOURCE,
    HF_DATASET_HANDLE,
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
    QualityGateError,
    QTEAError,
    QTEABenchmarkError,
    QTEALoadError,
    QTEAPackError,
    QTEAPublishError,
    QTEASourceError,
    QTEAVerifyError,
    RecoveryError,
    RotationError,
    SourceContractError,
    TemplateError,
    TernaryError,
)
from .qtea import (
    QTEA,
    QTEAConfig,
    QTEALinear,
    TernaryGroup,
    calibrate_and_quantize_projection,
    find_ornith_projections,
    pack_layer,
    pack_ternary,
    unpack_layer,
    unpack_ternary,
)
from .qtea_hf import (
    audit_remote_qtea,
    benchmark_qtea_inference,
    load_qtea_full_model,
    pack_qtea_shards,
    publish_qtea_sharded,
    run_qtea_benchmark,
    validate_qtea_source,
    verify_qtea_shards,
)
from .export import (
    MIXED_PRECISION_FORMAT,
    QTEA_OUTCOME_CONVERTED,
    QTEA_OUTCOME_QUALITY_ACCEPTED,
    QTEA_OUTCOME_QUALITY_REJECTED,
    audit_qtea_conversion,
    export_mixed_precision_package,
    export_qtea_package,
    load_mixed_precision_package,
    verify_mixed_precision_package,
    verify_qtea_package,
)
from .qtea import (
    EXPECTED_BF16_COUNT,
    EXPECTED_INT4_COUNT,
    EXPECTED_PROJECTION_COUNT,
    EXPECTED_QTEA_COUNT,
    PROJECTION_PRECISION_POLICY,
    MixedPrecisionPlan,
    ProjectionTarget,
    convert_model_mixed_precision,
    discover_mixed_precision_targets,
)
from .quant4 import (
    CODEC_NAME,
    INT8_CODEC_NAME,
    Protected4BitLinear,
    QuantizedInt8Embedding,
    QuantizedInt8Linear,
    convert_vocab_int8,
    measure_vocab_int8_delta_ce,
)
from .recovery import (
    MixedPrecisionRecoveryConfig,
    TeacherTargetCache,
    capture_teacher_targets,
    compute_fused_chunked_kl_loss,
    compute_recovery_loss,
    configure_mixed_precision_recovery,
    train_mixed_precision_recovery,
)
from .modeling import (
    extract_text_causal_lm_checkpoint,
    load_text_causal_model,
)
from .fallback import (
    GGUF_FALLBACK_FORMAT,
    GGUF_PRESERVED_F16_PATTERNS,
    LLAMA_CPP_PINNED_COMMIT,
    Q4_FALLBACK_FORMAT,
    export_text_q4_k_m_gguf,
    verify_gguf_fallback,
    verify_llama_cpp_commit,
)
from .gates import (
    gate_ce_quality,
    gate_logits_parity,
    gate_top1_parity,
)
from .evaluation import (
    EvalCase,
    EvalResult,
    EvaluationReport,
    aggregate,
    evaluate_case,
    format_thinking_prompt,
    parse_ornith_tool_calls,
    run_acceptance_suite,
)

__version__ = "2.0.0"

__all__ = [
    # config
    "DEFAULT_MODEL_HANDLE",
    "HF_MODEL_HANDLE",
    "KAGGLE_MODEL_HANDLE",
    "DEFAULT_DATASET_SOURCE",
    "HF_DATASET_HANDLE",
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
    "SourceContractError",
    "QualityGateError",
    "QTEAError",
    "QTEASourceError",
    "QTEAPackError",
    "QTEAVerifyError",
    "QTEAPublishError",
    "QTEALoadError",
    "QTEABenchmarkError",
    # qtea
    "QTEA",
    "QTEAConfig",
    "QTEALinear",
    "TernaryGroup",
    "calibrate_and_quantize_projection",
    "find_ornith_projections",
    "pack_layer",
    "pack_ternary",
    "unpack_layer",
    "unpack_ternary",
    # qtea hf tooling
    "audit_remote_qtea",
    "validate_qtea_source",
    "pack_qtea_shards",
    "verify_qtea_shards",
    "publish_qtea_sharded",
    "load_qtea_full_model",
    "benchmark_qtea_inference",
    "run_qtea_benchmark",
    # qtea export
    "export_qtea_package",
    "verify_qtea_package",
    "audit_qtea_conversion",
    "QTEA_OUTCOME_CONVERTED",
    "QTEA_OUTCOME_QUALITY_ACCEPTED",
    "QTEA_OUTCOME_QUALITY_REJECTED",
    # mixed-precision policy & INT4 codec (Ticket 007 / ADV-12.1)
    "PROJECTION_PRECISION_POLICY",
    "EXPECTED_QTEA_COUNT",
    "EXPECTED_INT4_COUNT",
    "EXPECTED_BF16_COUNT",
    "EXPECTED_PROJECTION_COUNT",
    "ProjectionTarget",
    "MixedPrecisionPlan",
    "discover_mixed_precision_targets",
    "convert_model_mixed_precision",
    "CODEC_NAME",
    "INT8_CODEC_NAME",
    "Protected4BitLinear",
    "QuantizedInt8Embedding",
    "QuantizedInt8Linear",
    "convert_vocab_int8",
    "measure_vocab_int8_delta_ce",
    "MIXED_PRECISION_FORMAT",
    "export_mixed_precision_package",
    "verify_mixed_precision_package",
    "load_mixed_precision_package",
    "MixedPrecisionRecoveryConfig",
    "TeacherTargetCache",
    "configure_mixed_precision_recovery",
    "capture_teacher_targets",
    "compute_fused_chunked_kl_loss",
    "compute_recovery_loss",
    "train_mixed_precision_recovery",
    "extract_text_causal_lm_checkpoint",
    "load_text_causal_model",
    "Q4_FALLBACK_FORMAT",
    "GGUF_FALLBACK_FORMAT",
    "GGUF_PRESERVED_F16_PATTERNS",
    "LLAMA_CPP_PINNED_COMMIT",
    "verify_llama_cpp_commit",
    "export_text_q4_k_m_gguf",
    "verify_gguf_fallback",
    "gate_ce_quality",
    "gate_logits_parity",
    "gate_top1_parity",
    "EvalCase",
    "EvalResult",
    "EvaluationReport",
    "format_thinking_prompt",
    "parse_ornith_tool_calls",
    "evaluate_case",
    "aggregate",
    "run_acceptance_suite",
    # controller
    "ExecutionController",
    "STAGE_ORDER",
    "STAGE_DONE",
    "CELL_TO_STAGE",
]
