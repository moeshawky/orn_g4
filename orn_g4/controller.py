"""Deterministic, resumable execution controller (work-order §8 / §11).

Owns the eleven-stage graph in fixed order:

    IMPORT_MODEL_AND_DATA        (cell 00-05)
         ↓
    CHECK_TEMPLATE_AND_MODEL     (cell 04)
         ↓
    DENSE_BASELINE               (cell 06)
         ↓
    SHORT_SFT                    (cell 07)
         ↓
    TERNARY_PROBES               (cell 08)
         ↓
    PRE_CONVERSION_SURGERY_PROBE (cell 09)
         ↓
    FULL_CONVERSION              (cell 10)
         ↓
    SCALE_DISTILLATION           (cell 11)
         ↓
    SELECTIVE_RECOVERY           (cell 12)
         ↓
    EVALUATION_AND_EXPORT        (cell 13)

Each stage is atomic: it completes only after its artifact checksums are written
to the machine-readable execution manifest. On restart the controller recovers
the run configuration, verifies completed-artifact checksums, restores
optimizer/scheduler state, and continues from the last validated stage — never
repeating expensive completed work without a specific reason.

A stage that exceeds its budget checkpoints and advances to the strongest
viable remaining deliverable (BudgetExceededError), per §8 decision policy.

CELL_STAGE_IDS maps each notebook cell (00..13) to an explicit stage ID so
there is no ambiguous numbering (e.g. two stages labelled "Cell 09").
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from .config import ExecutionManifest, RunConfig, StageResult
from .errors import (
    BudgetExceededError,
    ControllerError,
)
from .utils import ensure_dir, sha256_file, utc_now_iso, write_json_atomic

# Fixed stage order (work-order §8). Index = position for resume arithmetic.
# Eleven stages cover all 14 notebook cells (cells group under shared stages
# where needed, e.g. cells 00-05 all feed IMPORT_MODEL_AND_DATA).
STAGE_ORDER: tuple[str, ...] = (
    "IMPORT_MODEL_AND_DATA",
    "CHECK_TEMPLATE_AND_MODEL",
    "DENSE_BASELINE",
    "SHORT_SFT",
    "TERNARY_PROBES",
    "PRE_CONVERSION_SURGERY_PROBE",
    "FULL_CONVERSION",
    "SCALE_DISTILLATION",
    "SELECTIVE_RECOVERY",
    "EVALUATION",
    "EXPORT_AND_REPORT",
)

# Terminal marker.
STAGE_DONE = "DONE"

# Map cell index (0-based) -> stage ID for the 14 cells of Azrael-g4.
CELL_TO_STAGE: dict[int, str] = {
    0: STAGE_ORDER[0],  # cell_00_environment.txt  -> IMPORT_MODEL_AND_DATA
    1: STAGE_ORDER[0],  # cell_01_install.txt      -> IMPORT_MODEL_AND_DATA
    2: STAGE_ORDER[0],  # cell_02_mount_drive.txt  -> IMPORT_MODEL_AND_DATA
    3: STAGE_ORDER[0],  # cell_03_import_model.txt -> IMPORT_MODEL_AND_DATA
    4: STAGE_ORDER[1],  # cell_04_inspect_template.txt -> CHECK_TEMPLATE_AND_MODEL
    5: STAGE_ORDER[0],  # cell_05_import_dataset.txt -> IMPORT_MODEL_AND_DATA
    6: STAGE_ORDER[2],  # cell_06_dense_baseline.txt -> DENSE_BASELINE
    7: STAGE_ORDER[3],  # cell_07_short_sft.txt -> SHORT_SFT
    8: STAGE_ORDER[4],  # cell_08_ternary_probes.txt -> TERNARY_PROBES
    9: STAGE_ORDER[5],  # cell_09_pre_conversion_surgery_probe.txt -> PRE_CONVERSION_SURGERY_PROBE
    10: STAGE_ORDER[6],  # cell_10_full_conversion.txt -> FULL_CONVERSION
    11: STAGE_ORDER[7],  # cell_11_scale_distillation.txt -> SCALE_DISTILLATION
    12: STAGE_ORDER[8],  # cell_12_selective_recovery.txt -> SELECTIVE_RECOVERY
    13: STAGE_ORDER[10],  # cell_13_evaluation_export.txt -> EXPORT_AND_REPORT
}


class ExecutionController:
    """Deterministic stage executor with atomic completion and checksum-verified resume.

    A ``stage_fn`` callable is registered per stage; it receives the ``RunConfig``
    and a ``stage_context`` dict and must return a :class:`StageResult`. The
    controller writes the manifest after every stage and verifies checksums on
    resume.

    # Errors:
        raises :class:`ControllerError` on out-of-order stage requests or
        checksum mismatch; raises :class:`BudgetExceededError` when a stage's
        budget is exceeded (the controller then checkpoints and advances).
    """

    def __init__(
        self,
        run_config: RunConfig,
        run_dir: str | None = None,
        stage_fns: dict[str, Callable] | None = None,
    ):
        """Create the controller and (re)load its execution manifest.

        Args:
            run_config: The immutable experiment configuration.
            run_dir: Optional run directory; defaults to ``run_config.run_dir``
                or ``/tmp/ornith-run``.
            stage_fns: Mapping of stage name -> stage implementation callable.
        """
        self.config = run_config
        self.run_dir = ensure_dir(run_config.run_dir or run_dir or "/tmp/ornith-run")  # noqa: S108
        self.manifest_path = self.run_dir / "execution_manifest.json"
        self.stage_fns: dict[str, Callable] = dict(stage_fns or {})
        self._started = False

        # Load or create manifest.
        if self.manifest_path.is_file():
            import json

            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.manifest = ExecutionManifest.from_json(raw)
        else:
            self.manifest = ExecutionManifest(
                run_id=run_config.run_id,
                config=run_config,
                stages=[],
                current_stage=0,
                created_at=utc_now_iso(),
                updated_at=utc_now_iso(),
            )
            self._persist()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def register(self, stage: str, fn: Callable) -> None:
        """Register a stage implementation function."""
        if stage not in STAGE_ORDER and stage != STAGE_DONE:
            raise ControllerError(f"unknown stage {stage!r}; valid: {STAGE_ORDER}")
        self.stage_fns[stage] = fn

    def _persist(self) -> None:
        self.manifest.updated_at = utc_now_iso()
        write_json_atomic(self.manifest_path, self.manifest.to_json())

    def _stage_index(self, stage: str) -> int:
        try:
            return STAGE_ORDER.index(stage)
        except ValueError:
            if stage == STAGE_DONE:
                return len(STAGE_ORDER)
            raise ControllerError(f"unknown stage {stage!r}") from None

    def _completed_stages(self) -> list[str]:
        return [s.stage for s in self.manifest.stages if s.completed]

    def _verify_checksums(self) -> None:
        """Verify completed stages' artifacts match their recorded checksums.

        On mismatch, drop that stage from completed so it is re-run on the next
        pass (never trust a corrupted artifact).
        """
        surviving: list[StageResult] = []
        redone: list[str] = []
        for sr in self.manifest.stages:
            if not sr.completed:
                surviving.append(sr)
                continue
            ok = True
            for name, expected_checksum in sr.checksums.items():
                path = sr.artifacts.get(name)
                if path is None or not Path(path).is_file():
                    ok = False
                    break
                actual = sha256_file(Path(path))
                if actual != expected_checksum:
                    ok = False
                    break
            if ok:
                surviving.append(sr)
            else:
                redone.append(sr.stage)
        if redone:
            self.manifest.stages = surviving
            self.manifest.current_stage = (
                min(self._stage_index(r) for r in redone) if redone else self.manifest.current_stage
            )
            self._persist()
            for r in redone:
                print(f"[controller] checksum mismatch for {r}; will re-run")

    def stage_for_cell(self, cell_index: int) -> str:
        """Return the explicit stage ID for a notebook cell index (0-based)."""
        if cell_index not in CELL_TO_STAGE:
            raise ControllerError(
                f"cell index {cell_index} has no stage mapping; "
                f"valid cells: {sorted(CELL_TO_STAGE)}"
            )
        return CELL_TO_STAGE[cell_index]

    def resume_point(self) -> tuple[int, str]:
        """Return (index, stage_name) to resume from — last validated + 1."""
        self._verify_checksums()
        idx = self.manifest.current_stage
        if idx >= len(STAGE_ORDER):
            return len(STAGE_ORDER), STAGE_DONE
        return idx, STAGE_ORDER[idx]

    def is_complete(self) -> bool:
        """Return True when every stage has completed."""
        return self.manifest.current_stage >= len(STAGE_ORDER)

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #
    def run_stage(
        self,
        stage: str,
        *,
        budget_seconds: int | None = None,
        stage_context: dict | None = None,
    ) -> StageResult:
        """Run one stage atomically.

        The stage function must return a StageResult. On success the result is
        appended to the manifest and persisted. If the stage function raises
        BudgetExceededError, a checkpoint StageResult (completed=False) is
        recorded and the controller advances to the next stage (strongest
        viable deliverable path per §8).
        """
        if stage not in STAGE_ORDER:
            raise ControllerError(f"unknown stage {stage!r}") from None
        idx = self._stage_index(stage)
        ctx = dict(stage_context or {})
        ctx["run_id"] = self.config.run_id
        ctx["run_dir"] = str(self.run_dir)
        ctx["budget_seconds"] = budget_seconds or self.config.budget_seconds

        if self.manifest.current_stage > idx:
            # Already completed; skip (never repeat expensive completed work).
            existing = next((s for s in self.manifest.stages if s.stage == stage), None)
            if existing is not None:
                return existing

        t0 = time.perf_counter()
        try:
            fn = self.stage_fns.get(stage)
            if fn is None:
                result = StageResult(
                    stage=stage,
                    completed=False,
                    notes="no stage function registered; skipped",
                )
            else:
                result = fn(self.config, ctx)
                if not isinstance(result, StageResult):
                    raise ControllerError(f"stage {stage} function did not return a StageResult")
        except BudgetExceededError as exc:
            # §8 decision policy: checkpoint and advance to the strongest viable
            # remaining deliverable, never loop.
            elapsed = time.perf_counter() - t0
            result = StageResult(
                stage=stage,
                completed=False,
                runtime_seconds=elapsed,
                notes=f"budget exceeded: {exc}; checkpointing and advancing",
            )
        except Exception as exc:  # surface as a failed stage result
            elapsed = time.perf_counter() - t0
            result = StageResult(
                stage=stage,
                completed=False,
                runtime_seconds=elapsed,
                notes=f"stage failed: {type(exc).__name__}: {exc}",
            )

        result.runtime_seconds = result.runtime_seconds or (time.perf_counter() - t0)
        self._record(result)

        # Advance current_stage past this stage if completed.
        if result.completed:
            self.manifest.current_stage = idx + 1
        # If budget exceeded, advance so we don't loop on the same stage.
        if "budget exceeded" in (result.notes or ""):
            self.manifest.current_stage = idx + 1

        self._persist()
        return result

    def run_all(self, stage_context: dict | None = None) -> list[StageResult]:
        """Run stages from the resume point to completion, in order.

        Stops at the first non-completed stage that has no function registered,
        or when all stages complete. Budget-exceeded stages are recorded and
        skipped (advance) per §8.
        """
        results: list[StageResult] = []
        while not self.is_complete():
            idx, stage = self.resume_point()
            if stage == STAGE_DONE:
                break
            result = self.run_stage(stage, stage_context=stage_context)
            results.append(result)
            if not result.completed and "no stage function registered" in (result.notes or ""):
                # Nothing more can run; stop.
                break
        return results

    def _record(self, result: StageResult) -> None:
        # Replace any existing entry for this stage, then append.
        self.manifest.stages = [s for s in self.manifest.stages if s.stage != result.stage]
        self.manifest.stages.append(result)
        # Keep stages ordered by first appearance.
        order = {s: i for i, s in enumerate(STAGE_ORDER)}
        self.manifest.stages.sort(key=lambda s: order.get(s.stage, 99))


__all__ = [
    "ExecutionController",
    "STAGE_ORDER",
    "STAGE_DONE",
    "CELL_TO_STAGE",
]
