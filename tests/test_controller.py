"""Unit tests for the ExecutionController (work-order §8 / §11).

Covers: fixed stage order, atomic stage completion, resume never repeats a
completed stage, checksum mismatch triggers re-run, budget exceeded advances.
"""

import hashlib

from orn_g4.config import RunConfig, StageResult
from orn_g4.controller import STAGE_DONE, STAGE_ORDER, ExecutionController
from orn_g4.errors import BudgetExceededError


def _make_controller(tmp_path, run_id="test-run"):
    cfg = RunConfig(run_id=run_id, run_dir=str(tmp_path / "run"), budget_seconds=60)
    return ExecutionController(cfg)


def test_stage_order_is_fixed_and_complete():
    assert STAGE_ORDER[0] == "IMPORT_MODEL_AND_DATA"
    assert STAGE_ORDER[-1] == "EXPORT_AND_REPORT"
    assert len(STAGE_ORDER) == 11, (
        f"expected 11 stages (incl. PRE_CONVERSION_SURGERY_PROBE), got {len(STAGE_ORDER)}"
    )
    assert STAGE_DONE == "DONE"
    assert "PRE_CONVERSION_SURGERY_PROBE" in STAGE_ORDER


def test_run_stage_records_and_resume_skips_completed(tmp_path):
    ctl = _make_controller(tmp_path)
    calls = []

    def fn_a(config, ctx):
        calls.append("a")
        return StageResult(
            stage="IMPORT_MODEL_AND_DATA", completed=True, artifacts={}, checksums={}, notes="done"
        )

    ctl.register("IMPORT_MODEL_AND_DATA", fn_a)
    ctl.run_stage("IMPORT_MODEL_AND_DATA")
    assert calls == ["a"]

    # Resume: new controller on same run dir sees stage 0 completed.
    ctl2 = _make_controller(tmp_path)
    idx, stage = ctl2.resume_point()
    assert stage == "CHECK_TEMPLATE_AND_MODEL", f"expected stage 1, got {stage}"
    assert calls == ["a"], "completed stage must not be re-run"


def test_run_all_advances_through_registered_stages(tmp_path):
    ctl = _make_controller(tmp_path)
    order = []

    def make_fn(name):
        def fn(config, ctx):
            order.append(name)
            return StageResult(stage=name, completed=True, artifacts={}, checksums={})

        return fn

    for s in STAGE_ORDER[:3]:
        ctl.register(s, make_fn(s))
    results = ctl.run_all()
    assert order == list(STAGE_ORDER[:3])
    assert ctl.is_complete() is False  # stages 3..9 not registered


def test_budget_exceeded_advances_without_loop(tmp_path):
    ctl = _make_controller(tmp_path)
    calls = []

    def fn_budget(config, ctx):
        calls.append("budget_stage")
        raise BudgetExceededError("time limit")

    ctl.register("IMPORT_MODEL_AND_DATA", fn_budget)
    ctl.register(
        "CHECK_TEMPLATE_AND_MODEL",
        lambda config, ctx: StageResult(
            stage="CHECK_TEMPLATE_AND_MODEL", completed=True, artifacts={}, checksums={}
        ),
    )
    results = ctl.run_all()
    assert calls == ["budget_stage"]
    assert any(r.stage == "CHECK_TEMPLATE_AND_MODEL" and r.completed for r in results)


def test_checksum_mismatch_triggers_rerun(tmp_path):
    ctl = _make_controller(tmp_path)
    artifact = tmp_path / "art.txt"
    artifact.write_text("original")
    chk = "sha256:" + hashlib.sha256(b"original").hexdigest()
    r1 = StageResult(
        stage="IMPORT_MODEL_AND_DATA",
        completed=True,
        artifacts={"art.txt": str(artifact)},
        checksums={"art.txt": chk},
    )
    ctl.manifest.stages = [r1]
    ctl.manifest.current_stage = 1
    ctl._persist()

    # Corrupt the artifact -> new controller detects mismatch, re-runs.
    artifact.write_text("corrupted")
    ctl2 = _make_controller(tmp_path)
    idx, stage = ctl2.resume_point()
    assert stage == "IMPORT_MODEL_AND_DATA", f"expected re-run, got {stage}"
