"""Experiment report generation (work-order §10-E).

Records: actual GPU used, memory consumption, runtime, dataset and template
provenance, selected SFT checkpoint, quantization method comparisons, compression
coverage, effective bits per weight, recovery gains and regressions, exported
artifacts and checksums, known failures and limitations.

Every field that could not be observed in this build environment is labelled
UNVERIFIED-IN-THIS-ENVIRONMENT rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .env import EnvironmentReport, summarize_env

if TYPE_CHECKING:
    from .recovery import RecoveryReport


@dataclass
class ExperimentReport:
    """Complete experiment report (markdown-renderable)."""

    run_id: str
    environment: EnvironmentReport
    runtime_seconds: float
    peak_memory_bytes: int | None
    dataset_provenance: dict = field(default_factory=dict)
    template_provenance: dict = field(default_factory=dict)
    selected_sft_checkpoint: str | None = None
    sft_trainable_params: int | None = None
    sft_projection_coverage: dict = field(default_factory=dict)
    quantizer_comparison: dict = field(default_factory=dict)  # STE vs GPTQ results
    compression_coverage: dict = field(default_factory=dict)  # tensor -> ternary/except
    effective_bits_per_weight: float | None = None
    recovery_report: RecoveryReport | None = None
    exported_artifacts: dict = field(default_factory=dict)  # name -> path + checksum
    known_failures: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    unverified_items: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the experiment report as markdown."""
        lines: list[str] = []
        lines.append(f"# Experiment Report — run {self.run_id}")
        lines.append("")
        lines.append("## Compute environment")
        lines.append(f"- Runtime: {self.runtime_seconds:.1f} s")
        lines.append(f"- Peak memory: {self.peak_memory_bytes or 'unmeasured'}")
        lines.append(f"- GPU: {summarize_env(self.environment)}")
        for note in self.environment.notes:
            lines.append(f"  - {note}")
        lines.append("")
        lines.append("## Data and template provenance")
        zip_path = self.dataset_provenance.get("zip_path", "unrecorded")
        shard_count = self.dataset_provenance.get("shard_count", "unrecorded")
        lines.append(f"- Dataset ZIP: {zip_path}")
        lines.append(f"- Manifest shards resolved: {shard_count}")
        lines.append(f"- Template handle: {self.template_provenance.get('handle', 'unrecorded')}")
        template_hash = self.template_provenance.get("hash_sha256", "unrecorded")
        lines.append(f"- Template version/hash: {template_hash}")
        lines.append("")
        lines.append("## Selected SFT checkpoint")
        lines.append(f"- Checkpoint: {self.selected_sft_checkpoint or 'none selected'}")
        lines.append(f"- Trainable params: {self.sft_trainable_params or 'unmeasured'}")
        lines.append(f"- Projection coverage: {self.sft_projection_coverage or {}}")
        lines.append("")
        lines.append("## Quantizer method comparison")
        for method, res in self.quantizer_comparison.items():
            lines.append(f"- {method}: {res}")
        lines.append("")
        lines.append("## Compression")
        lines.append(f"- Effective bits/weight: {self.effective_bits_per_weight or 'unmeasured'}")
        lines.append(f"- Coverage: {self.compression_coverage or {}}")
        lines.append("")
        lines.append("## Recovery")
        if self.recovery_report is not None:
            rr = self.recovery_report
            lines.append(f"- Scale distillation converged: {rr.scale_distillation.converged}")
            residual_ranks = [r.rank for r in rr.residuals.values()]
            lines.append(f"- Residual ranks tested: {residual_ranks}")
            lines.append(f"- Overall improved: {rr.overall_improved}")
            skipped = rr.recurrent_refinement_skipped
            reason = rr.recurrent_reason
            lines.append(f"- Recurrent refinement skipped: {skipped} ({reason})")
        lines.append("")
        lines.append("## Exported artifacts")
        for name, info in self.exported_artifacts.items():
            lines.append(f"- {name}: {info}")
        lines.append("")
        lines.append("## Known failures and limitations")
        for failure in self.known_failures:
            lines.append(f"- {failure}")
        for limitation in self.limitations:
            lines.append(f"- {limitation}")
        if self.unverified_items:
            lines.append("")
            lines.append("## Unverified (this build environment)")
            for u in self.unverified_items:
                lines.append(f"- {u}")
        return "\n".join(lines) + "\n"


def build_unverified_list(env: EnvironmentReport, labels: list[str]) -> list[str]:
    """Combine environment notes with explicit unverified labels."""
    items = list(env.notes)
    items.extend(labels)
    return items


__all__ = [
    "ExperimentReport",
    "build_unverified_list",
]
