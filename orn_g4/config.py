"""Typed configuration dataclasses for the Ornith-1.5-9b ternary campaign.

Every field is a concrete value with a documented source: nothing is guessed
from a model name, a platform, or a framework default. These dataclasses are the
single contract between the notebook cells (orchestration) and the ``orn_g4``
modules (implementation). Serialize to/from JSON for the run manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

#: Kaggle model handle, exact version, as mandated by work-order §1.
KAGGLE_MODEL_HANDLE = "mohamedshawky4340/ornith-1.5-9b/transformers/bf16/1"

#: Native template repository (authoritative cross-check, work-order §2).
TEMPLATE_REPO = "https://huggingface.co/moeshawky/Azrael-9b-chat-template"

#: Google Drive root path for the dataset ZIP (work-order §3).
DRIVE_DATASET_ZIP = "/content/drive/MyDrive/azrael-dataset.zip"

#: Persistent run directory root on Google Drive (work-order §7 / §11).
DRIVE_RUN_ROOT = "/content/drive/MyDrive/ornith-ternary-runs"

#: Frozen pre-tokenized SFT dataset release figures (work-order §1).
EXPECTED_DATASET_RECORDS = 17_848
EXPECTED_DATASET_SHARDS = 7
EXPECTED_TOTAL_TOKENS = 30_029_360
EXPECTED_SUPERVISED_TOKENS = 8_748_755
EXPECTED_MAX_SEQ_LEN = 8_192

#: Expected EOS token ID in tokenizer contract
EXPECTED_EOS_ID = 248046

#: Expected template hash prefix and revision
EXPECTED_TEMPLATE_SHA256_PREFIX = "9dd2fbd2"
EXPECTED_TEMPLATE_REVISION = "33e8951"

#: Explicit stage IDs for every notebook cell (work-order §1: no ambiguous numbering).
CELL_STAGE_IDS: tuple[str, ...] = (
    "STAGE_00_ENV_RESOLVE",
    "STAGE_01_RUNTIME_DEPS",
    "STAGE_02_MOUNT_DRIVE",
    "STAGE_03_IMPORT_MODEL",
    "STAGE_04_INSPECT_TEMPLATE",
    "STAGE_05_IMPORT_DATASET",
    "STAGE_06_DENSE_BASELINE",
    "STAGE_07_BF16_SFT",
    "STAGE_08_TERNARY_PROBES",
    "STAGE_09_SURGERY_PROBES",
    "STAGE_10_FULL_CONVERSION",
    "STAGE_11_SCALE_RECOVERY",
    "STAGE_12_SELECTIVE_RECOVERY",
    "STAGE_13_EVALUATION_EXPORT",
)


@dataclass(frozen=True)
class RunDirectoryLayout:
    """Persistent, versioned run directory layout (work-order §7).

    Organizes all checkpoints, exported artifacts, evaluations, and reports
    under one authoritative ``runs/<run_id>/`` root so notebook cells never
    collide or scatter state across ad-hoc directories.
    """

    root: Path

    @classmethod
    def from_run_dir(cls, run_dir: Path | str, run_id: str | None = None) -> RunDirectoryLayout:
        """Create and initialize the directory tree under ``run_dir``."""
        base = Path(run_dir).resolve()
        layout = cls(root=base)
        layout.ensure_all()
        return layout

    @property
    def manifest_path(self) -> Path:
        """Path to the run manifest JSON file."""
        return self.root / "manifest.json"

    @property
    def execution_manifest_path(self) -> Path:
        """Path to the execution manifest JSON file."""
        return self.root / "execution_manifest.json"

    @property
    def environment_path(self) -> Path:
        """Path to the environment report JSON file."""
        return self.root / "environment.json"

    @property
    def dataset_manifest_path(self) -> Path:
        """Path to the dataset manifest JSON file."""
        return self.root / "dataset_manifest.json"

    @property
    def source_manifest_path(self) -> Path:
        """Path to the source manifest JSON file."""
        return self.root / "source_manifest.json"

    @property
    def checkpoints_dir(self) -> Path:
        """Directory containing stage checkpoints."""
        return self.root / "checkpoints"

    @property
    def sft_checkpoint_dir(self) -> Path:
        """Directory for SFT stage checkpoints."""
        return self.checkpoints_dir / "sft"

    @property
    def probes_checkpoint_dir(self) -> Path:
        """Directory for ternary probe checkpoints."""
        return self.checkpoints_dir / "probes"

    @property
    def surgery_checkpoint_dir(self) -> Path:
        """Directory for surgery probe checkpoints."""
        return self.checkpoints_dir / "surgery"

    @property
    def conversion_checkpoint_dir(self) -> Path:
        """Directory for ternary conversion checkpoints."""
        return self.checkpoints_dir / "conversion"

    @property
    def scale_recovery_checkpoint_dir(self) -> Path:
        """Directory for scale recovery checkpoints."""
        return self.checkpoints_dir / "scale_recovery"

    @property
    def selective_qat_checkpoint_dir(self) -> Path:
        """Directory for selective QAT checkpoints."""
        return self.checkpoints_dir / "selective_qat"

    @property
    def artifacts_dir(self) -> Path:
        """Directory containing exported artifacts."""
        return self.root / "artifacts"

    @property
    def bf16_reference_dir(self) -> Path:
        """Directory for Artifact A (BF16 adaptation)."""
        return self.artifacts_dir / "bf16_reference"

    @property
    def ternary_canonical_dir(self) -> Path:
        """Directory for Artifact B (canonical ternary checkpoint)."""
        return self.artifacts_dir / "ternary_canonical"

    @property
    def trainable_continuation_dir(self) -> Path:
        """Directory for Artifact C (trainable continuation)."""
        return self.artifacts_dir / "trainable_continuation"

    @property
    def dense_interoperability_dir(self) -> Path:
        """Directory for dense interoperability export."""
        return self.artifacts_dir / "dense_interoperability"

    @property
    def gguf_dir(self) -> Path:
        """Directory for Artifact D (GGUF/PTQ1_0)."""
        return self.artifacts_dir / "gguf"

    @property
    def evaluation_dir(self) -> Path:
        """Directory containing evaluation results."""
        return self.root / "evaluation"

    @property
    def reports_dir(self) -> Path:
        """Directory containing experiment reports."""
        return self.root / "reports"

    def ensure_all(self) -> None:
        """Create all standard directories idempotently."""
        for d in (
            self.root,
            self.sft_checkpoint_dir,
            self.probes_checkpoint_dir,
            self.surgery_checkpoint_dir,
            self.conversion_checkpoint_dir,
            self.scale_recovery_checkpoint_dir,
            self.selective_qat_checkpoint_dir,
            self.bf16_reference_dir,
            self.ternary_canonical_dir,
            self.trainable_continuation_dir,
            self.trainable_continuation_dir / "optimizer_state",
            self.dense_interoperability_dir,
            self.gguf_dir,
            self.evaluation_dir,
            self.reports_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunConfig:
    """Immutable experiment configuration captured once at run start.

    Attributes:
        run_id: Machine-readable run identifier (timestamp + short hash).
        model_handle: Kaggle model handle (verbatim, recorded for provenance).
        template_repo: Template repository URL for cross-check.
        dataset_zip: Absolute path of the dataset ZIP on Google Drive.
        run_dir: Persistent run directory under DRIVE_RUN_ROOT/<run_id>/.
        local_work_dir: Colab-local scratch for active training/temp data.
        precision: BF16 (work-order §5).
        lora_rank: LoRA rank (32).
        lora_alpha: LoRA alpha (64).
        lora_lr: LoRA learning rate (2e-5).
        lora_microbatch: initial microbatch (1).
        supervised_token_budget: initial SFT phase budget (~2M supervised tokens).
        corpus_supervised_tokens: full dataset supervised token count (8,748,755).
        scale_recovery_token_budget: explicit token budget for scale recovery.
        selective_qat_token_budget: explicit token budget for selective QAT.
        continuation_schedule: cumulative supervised-token milestones for
            continuing into the remaining 8.75M corpus without resetting run identity.
        reference_kl_weight: optional weight for reference-matching loss (0.0
            means disabled; never a mandatory full-logit KL dependency per §5).
        ternary_group: input features per output row (128).
        residual_ranks: low-rank residual ranks tested (16, 32).
        qat_rank: selective QAT rank when engaged.
        max_seq_len: bucketed sequence-length ceiling (8192).
        hessian_budget_bytes: runtime-observed cap on per-projection Hessian
            accumulator bytes (chosen from free VRAM, never hardcoded 24 GB).
        budget_seconds: per-stage wall-clock budget (operator-set).
        seed: deterministic seed for all stochastic choices.
        schema_version: version of the run manifest schema.
    """

    run_id: str
    model_handle: str = KAGGLE_MODEL_HANDLE
    template_repo: str = TEMPLATE_REPO
    dataset_zip: str = DRIVE_DATASET_ZIP
    run_dir: str = ""
    local_work_dir: str = "/tmp/ornith-run"  # noqa: S108  (config default, not created here)
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_lr: float = 2e-5
    lora_microbatch: int = 1
    supervised_token_budget: int = 2_000_000
    corpus_supervised_tokens: int = EXPECTED_SUPERVISED_TOKENS
    scale_recovery_token_budget: int = 500_000
    selective_qat_token_budget: int = 500_000
    continuation_schedule: tuple[int, ...] = (
        2_000_000,
        4_000_000,
        6_000_000,
        EXPECTED_SUPERVISED_TOKENS,
    )
    reference_kl_weight: float = 0.0
    ternary_group: int = 128
    residual_ranks: tuple[int, ...] = (16, 32)
    qat_rank: int = 32
    max_seq_len: int = EXPECTED_MAX_SEQ_LEN
    hessian_budget_bytes: int = 0  # 0 means "derive from observed free VRAM"
    budget_seconds: int = 3600
    seed: int = 1337
    schema_version: str = "2.0.0"


@dataclass(frozen=True)
class Settings:
    """Operational settings matching the working G4 notebook execution contracts."""

    MODE: str = "SMOKE"  # 'STATIC' / 'SMOKE' / 'FULL'
    RUN_ID: str = "ornith-g4-v1"
    MODEL_HANDLE: str = KAGGLE_MODEL_HANDLE
    MODEL_OVERRIDE: str = ""
    DATASET_ZIP: str = DRIVE_DATASET_ZIP
    DATASET_DIR: str = ""
    RUN_ROOT: str = DRIVE_RUN_ROOT
    LOCAL_DATA: str = "/content/ornith-frozen-data"
    SHARD_GLOB: str = ""
    EXPECTED_FROZEN_SHARDS: int = EXPECTED_DATASET_SHARDS
    EXPECTED_FROZEN_RECORDS: int = EXPECTED_DATASET_RECORDS
    EXPECTED_UNIQUE_IDS: int = 17_658
    EXPECTED_TOTAL_TOKENS: int = EXPECTED_TOTAL_TOKENS
    EXPECTED_SUPERVISED_TOKENS: int = EXPECTED_SUPERVISED_TOKENS
    EXPECTED_EOS_ID: int = EXPECTED_EOS_ID
    EXPECTED_TEMPLATE_SHA256_PREFIX: str = EXPECTED_TEMPLATE_SHA256_PREFIX
    EXPECTED_TEMPLATE_REVISION: str = EXPECTED_TEMPLATE_REVISION
    IDS_KEY: str = "input_ids"
    MASK_KEY: str = "attention_mask"
    LABELS_KEY: str = "labels"
    GROUP: int = 128
    LORA_RANK: int = 32
    LORA_ALPHA: int = 64
    SFT_LR: float = 2e-5
    SFT_SUPERVISED_TOKENS: int = 2_000_000
    SCALE_RECOVERY_TOKENS: int = 1_000_000
    RESIDUAL_RECOVERY_TOKENS: int = 6_000_000
    QAT_TOKENS: int = 0
    QAT_MAX_SHADOW_PARAMS: int = 4_000_000
    RECOVERY_LR: float = 1e-4
    QAT_LR: float = 1e-5
    MAX_SEQ_LEN: int = EXPECTED_MAX_SEQ_LEN
    CALIB_MAX_SEQ: int = 512
    CALIB_EXAMPLES_PER_BLOCK: int = 8
    HELDOUT_EXAMPLES_PER_BLOCK: int = 4
    ACT_TOKENS_PER_MODULE: int = 256
    RESIDUAL_RANK: int = 16
    MAX_RESIDUAL_MODULES: int = 12
    EXCEPTION_ERROR_THRESHOLD: float = 0.65
    RUN_HADAMARD_DIAGNOSTIC: bool = True
    CHECKPOINT_EVERY: int = 50
    EVAL_EXAMPLES: int = 24
    DENSE_EXPORT: bool = True
    GGUF_EXPORT: bool = False
    SEED: int = 1337

    @property
    def sft_budget(self) -> int:
        return 2048 if self.MODE == "SMOKE" else self.SFT_SUPERVISED_TOKENS

    @property
    def scale_budget(self) -> int:
        return 1024 if self.MODE == "SMOKE" else self.SCALE_RECOVERY_TOKENS

    @property
    def resid_budget(self) -> int:
        return 1024 if self.MODE == "SMOKE" else self.RESIDUAL_RECOVERY_TOKENS

    @property
    def qat_budget(self) -> int:
        return 0 if self.MODE == "SMOKE" else self.QAT_TOKENS

    @property
    def calib_n(self) -> int:
        return 2 if self.MODE == "SMOKE" else self.CALIB_EXAMPLES_PER_BLOCK

    @property
    def held_n(self) -> int:
        return 1 if self.MODE == "SMOKE" else self.HELDOUT_EXAMPLES_PER_BLOCK


@dataclass
class StageResult:
    """Atomic result of one controller stage.

    Attributes:
        stage: stage name (one of the fixed STAGE_ORDER entries).
        completed: whether the stage reached a validated artifact.
        artifacts: mapping of artifact name -> absolute path.
        checksums: mapping of artifact name -> "sha256:<hex>".
        runtime_seconds: wall time consumed.
        notes: human-readable summary + any UNVERIFIED labels.
    """

    stage: str
    completed: bool
    artifacts: dict[str, str] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    notes: str = ""

    def to_json(self) -> dict:
        """Serialize this stage result to a JSON-serializable dict."""
        return {
            "stage": self.stage,
            "completed": self.completed,
            "artifacts": dict(self.artifacts),
            "checksums": dict(self.checksums),
            "runtime_seconds": self.runtime_seconds,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, d: dict) -> StageResult:
        """Rebuild a :class:`StageResult` from a JSON-serializable dict."""
        return cls(
            stage=d["stage"],
            completed=bool(d["completed"]),
            artifacts={str(k): str(v) for k, v in d.get("artifacts", {}).items()},
            checksums={str(k): str(v) for k, v in d.get("checksums", {}).items()},
            runtime_seconds=float(d.get("runtime_seconds", 0.0)),
            notes=str(d.get("notes", "")),
        )


@dataclass
class ExecutionManifest:
    """Machine-readable execution manifest persisted to the run directory.

    Attributes:
        run_id: the run this manifest belongs to.
        config: the RunConfig that started it.
        stages: ordered list of completed StageResults.
        current_stage: the stage to resume from (last validated + 1).
        created_at: ISO-8601 timestamp of manifest creation.
        updated_at: ISO-8601 timestamp of last mutation.
        cell_stage_ids: explicit ordered stage IDs for the 14 notebook cells.
    """

    run_id: str
    config: RunConfig
    stages: list[StageResult] = field(default_factory=list)
    current_stage: int = 0
    created_at: str = ""
    updated_at: str = ""
    cell_stage_ids: tuple[str, ...] = CELL_STAGE_IDS

    def to_json(self) -> dict:
        """Serialize this execution manifest to a JSON-serializable dict."""
        return {
            "schema_version": self.config.schema_version,
            "run_id": self.run_id,
            "cell_stage_ids": list(self.cell_stage_ids),
            "config": {
                "run_id": self.config.run_id,
                "model_handle": self.config.model_handle,
                "template_repo": self.config.template_repo,
                "dataset_zip": self.config.dataset_zip,
                "run_dir": self.config.run_dir,
                "local_work_dir": self.config.local_work_dir,
                "precision": self.config.precision,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "lora_lr": self.config.lora_lr,
                "lora_microbatch": self.config.lora_microbatch,
                "supervised_token_budget": self.config.supervised_token_budget,
                "corpus_supervised_tokens": self.config.corpus_supervised_tokens,
                "scale_recovery_token_budget": self.config.scale_recovery_token_budget,
                "selective_qat_token_budget": self.config.selective_qat_token_budget,
                "continuation_schedule": list(self.config.continuation_schedule),
                "reference_kl_weight": self.config.reference_kl_weight,
                "ternary_group": self.config.ternary_group,
                "residual_ranks": list(self.config.residual_ranks),
                "qat_rank": self.config.qat_rank,
                "max_seq_len": self.config.max_seq_len,
                "hessian_budget_bytes": self.config.hessian_budget_bytes,
                "budget_seconds": self.config.budget_seconds,
                "seed": self.config.seed,
                "schema_version": self.config.schema_version,
            },
            "stages": [s.to_json() for s in self.stages],
            "current_stage": self.current_stage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, d: dict) -> ExecutionManifest:
        """Rebuild an :class:`ExecutionManifest` from a JSON-serializable dict."""
        cfg_fields = {k: v for k, v in d["config"].items() if k in RunConfig.__dataclass_fields__}
        if "residual_ranks" in cfg_fields and isinstance(cfg_fields["residual_ranks"], list):
            cfg_fields["residual_ranks"] = tuple(int(x) for x in cfg_fields["residual_ranks"])
        if "continuation_schedule" in cfg_fields and isinstance(
            cfg_fields["continuation_schedule"], list
        ):
            cfg_fields["continuation_schedule"] = tuple(
                int(x) for x in cfg_fields["continuation_schedule"]
            )
        cfg = RunConfig(**cfg_fields)
        return cls(
            run_id=d["run_id"],
            config=cfg,
            stages=[StageResult.from_json(s) for s in d.get("stages", [])],
            current_stage=int(d.get("current_stage", 0)),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            cell_stage_ids=tuple(d.get("cell_stage_ids", CELL_STAGE_IDS)),
        )


@dataclass(frozen=True)
class TernaryArtifact:
    """The canonical effective ternary checkpoint contents (work-order §10-B).

    Attributes:
        codes: mapping tensor name -> int8 tensor in {-1, 0, +1}.
        scales: mapping tensor name -> fp32 group scale vector (positive).
        group_axes: mapping tensor name -> (group_axis, group_size) tuple.
        shapes: mapping tensor name -> original weight shape tuple.
        rotation: dict with rotation metadata (None when unrotated).
        precision_exceptions: list of tensor names kept in source precision.
        config: the source model config.json (parsed).
        template_fingerprint: dict with template handle + version/hash.
        provenance: dict with Kaggle handle, version, dataset manifest path.
    """

    codes: dict[str, object]  # torch.IntTensor at runtime; avoid import-time dep
    scales: dict[str, object]  # torch.FloatTensor
    group_axes: dict[str, tuple[int, int]]
    shapes: dict[str, tuple[int, ...]]
    rotation: dict | None = None
    precision_exceptions: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    template_fingerprint: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize this ternary artifact's metadata to a JSON-serializable dict."""
        return {
            "group_axes": {k: list(v) for k, v in self.group_axes.items()},
            "shapes": {k: list(v) for k, v in self.shapes.items()},
            "rotation": self.rotation,
            "precision_exceptions": list(self.precision_exceptions),
            "config": self.config,
            "template_fingerprint": self.template_fingerprint,
            "provenance": self.provenance,
            # codes/scales are tensors; serialized separately by export.py
        }


__all__ = [
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
    "RunDirectoryLayout",
    "RunConfig",
    "StageResult",
    "ExecutionManifest",
    "TernaryArtifact",
]
