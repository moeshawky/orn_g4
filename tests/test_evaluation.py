"""Tests for strict offline evaluation, XML tool parsing, coding execution, and __init__ exports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from orn_g4 import CODEC_NAME
import orn_g4
from orn_g4.errors import QualityGateError
from orn_g4.evaluation import (
    EvalCase,
    EvalResult,
    EvaluationReport,
    aggregate,
    evaluate_case,
    format_thinking_prompt,
    parse_ornith_tool_calls,
    run_acceptance_suite,
)


def test_xml_tool_call_parsing_and_negative_controls() -> None:
    good_output = (
        "Let me inspect the file first.\n</think>\n\n"
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\n/tmp/a.py\n</parameter>\n"
        "<parameter=limit>10</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    prompt = "Inspect /tmp/a.py\n<think>\n"
    ok, calls = parse_ornith_tool_calls(good_output, prompt=prompt)
    assert ok is True
    assert calls == [{"name": "read_file", "arguments": {"path": "/tmp/a.py", "limit": "10"}}]

    case = EvalCase(
        case_id="tool_ok",
        category="tool_selection",
        prompt="Inspect /tmp/a.py",
        expected_tool="read_file",
        expected_args={"path": "/tmp/a.py", "limit": 10},
        enable_thinking=True,
    )
    res = evaluate_case(case, lambda p, m: good_output)
    assert res.passed is True
    assert res.syntax_valid is True
    assert res.tool_selected == "read_file"
    assert res.args_match is True

    # Negative control 1: unescaped regex / prose trap must NOT pass.
    prose_bad = "You should call function: read_file with path /tmp/a.py and <tool_call is nice>"
    res_prose = evaluate_case(
        EvalCase(
            case_id="tool_prose",
            category="tool_selection",
            prompt="Inspect /tmp/a.py",
            expected_tool="read_file",
            expected_args={"path": "/tmp/a.py"},
        ),
        lambda p, m: prose_bad,
    )
    assert res_prose.passed is False
    assert res_prose.syntax_valid is False
    assert res_prose.tool_selected is None
    assert res_prose.args_match is False

    # Negative control 2: wrong tool name with matching args must NOT leave args_match=None / passed=True.
    wrong_tool = (
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>/tmp/a.py</parameter>\n"
        "</function>\n</tool_call>"
    )
    res_wrong_tool = evaluate_case(
        EvalCase(
            case_id="tool_wrong_name",
            category="tool_selection",
            prompt="Inspect /tmp/a.py",
            expected_tool="read_file",
            expected_args={"path": "/tmp/a.py"},
        ),
        lambda p, m: wrong_tool,
    )
    assert res_wrong_tool.syntax_valid is True
    assert res_wrong_tool.tool_selected == "write_file"
    assert res_wrong_tool.args_match is False
    assert res_wrong_tool.passed is False

    # Negative control 3: wrong parameter value (substring trap) must fail.
    wrong_arg = (
        "<tool_call>\n<function=read_file>\n"
        "<parameter=path>/tmp/a.py.bak</parameter>\n"
        "<parameter=limit>20</parameter>\n"
        "</function>\n</tool_call>"
    )
    res_wrong_arg = evaluate_case(
        EvalCase(
            case_id="tool_wrong_arg",
            category="tool_selection",
            prompt="Inspect /tmp/a.py",
            expected_tool="read_file",
            expected_args={"path": "/tmp/a.py", "limit": 10},
        ),
        lambda p, m: wrong_arg,
    )
    assert res_wrong_arg.tool_selected == "read_file"
    assert res_wrong_arg.args_match is False
    assert res_wrong_arg.passed is False

    # Negative control 4: tool call placed inside <think>...</think> only must NOT count.
    hidden_inside_think = (
        "<think><tool_call><function=read_file><parameter=path>/tmp/a.py</parameter></function></tool_call></think>\n"
        "No tool needed."
    )
    res_hidden = evaluate_case(
        EvalCase(
            case_id="tool_hidden_in_think",
            category="tool_selection",
            prompt="Inspect /tmp/a.py",
            expected_tool="read_file",
            expected_args={"path": "/tmp/a.py"},
        ),
        lambda p, m: hidden_inside_think,
    )
    assert res_hidden.passed is False
    assert res_hidden.syntax_valid is False


def test_thinking_on_prompt_contract_and_executable_coding_assertions() -> None:
    seen_prompts: list[str] = []

    class _FakeTokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool, enable_thinking: bool
        ) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            tag = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
            return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n{tag}"

    tok = _FakeTokenizer()
    rendered = format_thinking_prompt("Implement add", tokenizer=tok, enable_thinking=True)
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")

    coding_pass = EvalCase(
        case_id="code_pass",
        category="coding",
        prompt="Write def add(a, b) returning sum.",
        test_code="assert add(2, 3) == 5\nassert add(-4, 4) == 0",
        enable_thinking=True,
    )
    coding_fail_logic = EvalCase(
        case_id="code_wrong_logic",
        category="coding",
        prompt="Write def sub(a, b) returning difference.",
        test_code="assert sub(5, 2) == 3",
        enable_thinking=True,
    )
    coding_syntax_err = EvalCase(
        case_id="code_syntax_err",
        category="coding",
        prompt="Write def bad().",
        test_code="assert True",
        enable_thinking=True,
    )
    coding_missing_think_close = EvalCase(
        case_id="code_missing_think",
        category="coding",
        prompt="Write def ok(): return 1",
        test_code="assert ok() == 1",
        enable_thinking=True,
    )

    def _gen(prompt: str, max_tokens: int) -> str:
        seen_prompts.append(prompt)
        assert prompt.count("<think>") > prompt.count("</think>")
        if "def add" in prompt:
            return "Thinking: return a + b.\n</think>\n\n```python\ndef add(a, b):\n    return a + b\n```"
        if "def sub" in prompt:
            return "Thinking: oops a + b.\n</think>\n\n```python\ndef sub(a, b):\n    return a + b\n```"
        if "def bad" in prompt:
            return "Thinking: broken syntax.\n</think>\n\n```python\ndef bad(:\n```"
        return "```python\ndef ok():\n    return 1\n```"  # missing </think>!

    report = run_acceptance_suite(
        [coding_pass, coding_fail_logic, coding_syntax_err, coding_missing_think_close],
        _gen,
        split="probe",
        tokenizer=tok,
    )
    assert len(seen_prompts) == 4
    by_id = {r.case_id: r for r in report.cases}
    assert by_id["code_pass"].passed is True
    assert by_id["code_pass"].thinking_valid is True
    assert by_id["code_pass"].code_passed is True

    assert by_id["code_wrong_logic"].syntax_valid is True
    assert by_id["code_wrong_logic"].code_passed is False
    assert by_id["code_wrong_logic"].passed is False

    assert by_id["code_syntax_err"].syntax_valid is False
    assert by_id["code_syntax_err"].code_passed is False
    assert by_id["code_syntax_err"].passed is False

    assert by_id["code_missing_think"].thinking_valid is False
    assert by_id["code_missing_think"].passed is False
    assert report.reasoning_preserved is False
    assert report.coding_correctness == pytest.approx(0.25)

    # Pre-rendered thinking-OFF prompt passed to enable_thinking=True case must fail thinking contract.
    bad_off_prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    off_case = EvalCase(
        case_id="prompt_off_mismatch",
        category="coding",
        prompt=bad_off_prompt,
        test_code="assert f() == 1",
        enable_thinking=True,
    )
    res_off = evaluate_case(off_case, lambda p, m: "</think>\n```python\ndef f(): return 1\n```")
    assert res_off.passed is False


def test_acceptance_suite_seals_heldout_and_enforces_quality_gates() -> None:
    case = EvalCase(
        case_id="c1",
        category="coding",
        prompt="Write idfn",
        test_code="assert idfn(7) == 7",
    )
    gen = lambda p, m: "```python\ndef idfn(x):\n    return x\n```"

    with pytest.raises(QualityGateError, match="sealed"):
        run_acceptance_suite([case], gen, split="heldout")

    heldout_case = EvalCase(
        case_id="c2",
        category="coding",
        prompt="Write idfn",
        test_code="assert idfn(7) == 7",
        split="eval_firewall",
    )
    with pytest.raises(QualityGateError, match="sealed"):
        run_acceptance_suite([heldout_case], gen, split="probe")

    with pytest.raises(QualityGateError, match="at least one"):
        run_acceptance_suite([], gen, split="probe")

    with pytest.raises(QualityGateError, match="CE degradation"):
        run_acceptance_suite(
            [case],
            gen,
            split="heldout",
            allow_heldout=True,
            ce_measurements={"teacher_ce": 1.5, "student_ce": 2.2},
            max_ce_regression=0.50,
        )

    rep = run_acceptance_suite(
        [case],
        gen,
        split="heldout",
        allow_heldout=True,
        ce_measurements={"teacher_ce": 1.5, "student_ce": 1.6},
        max_ce_regression=0.50,
        ref_logits=np.array([[2.0, 0.5]], dtype=np.float32),
        reload_logits=np.array([[2.001, 0.501]], dtype=np.float32),
    )
    assert rep.cases[0].passed is True


def test_package_reexports_and_cpu_no_torch_import_safety() -> None:
    expected_symbols = [
        "PROJECTION_PRECISION_POLICY",
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
        "configure_mixed_precision_recovery",
        "capture_teacher_targets",
        "compute_fused_chunked_kl_loss",
        "train_mixed_precision_recovery",
        "extract_text_causal_lm_checkpoint",
        "load_text_causal_model",
        "GGUF_FALLBACK_FORMAT",
        "LLAMA_CPP_PINNED_COMMIT",
        "export_text_q4_k_m_gguf",
        "verify_gguf_fallback",
        "gate_ce_quality",
        "gate_logits_parity",
        "gate_top1_parity",
        "EvalCase",
        "EvalResult",
        "EvaluationReport",
        "evaluate_case",
        "aggregate",
        "run_acceptance_suite",
    ]
    for sym in expected_symbols:
        assert hasattr(orn_g4, sym), f"Missing symbol in orn_g4: {sym}"
        assert sym in orn_g4.__all__, f"Missing symbol in orn_g4.__all__: {sym}"

    repo_root = str(Path(__file__).resolve().parents[1])
    code = (
        "import importlib.abc, sys\n"
        "class _BlockTorch(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
        "            raise ImportError('simulated no-torch host')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockTorch())\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        "ns = {}\n"
        "exec('from orn_g4 import *', ns)\n"
        "assert 'QTEALinear' in ns and 'Protected4BitLinear' in ns and 'run_acceptance_suite' in ns\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"no-torch import failed: {proc.stderr}\n{proc.stdout}"


def test_trace_replay_followup_closes_think_and_preserves_turn1() -> None:
    """Trace replay carries Turn-1 output, closes open </think>, and separates Turn-2 thinking."""
    class _FakeTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            enable_thinking: bool,
        ) -> str:
            assert tokenize is False and add_generation_prompt is True
            tag = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
            return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n{tag}"

    prompts_seen: list[str] = []

    def _gen_good(prompt: str, max_tokens: int) -> str:
        prompts_seen.append(prompt)
        if len(prompts_seen) == 1:
            return (
                "I should call read_file.\n</think>\n\n"
                "<tool_call>\n<function=read_file>\n<parameter=path>missing.txt</parameter>\n</function>\n</tool_call><|im_end|>"
            )
        assert "</tool_call>" in prompt
        assert "</think>\n<|im_end|>\n<|im_start|>user\nTOOL OBSERVATION:" in prompt or "<|im_end|>\n<|im_start|>user\nTOOL OBSERVATION:" in prompt
        assert prompt.count("<think>") > prompt.count("</think>")
        assert prompt.endswith("<|im_start|>assistant\n<think>\n")
        return "Check alternatives.\n</think>\n\nThe file missing.txt does not exist; switching to fallback strategy."

    case = EvalCase(
        case_id="rec_case",
        category="recovery",
        prompt="Inspect missing.txt and recover if absent.",
        tool_observation="Tool Error: FileNotFoundError",
        enable_thinking=True,
    )
    rep = run_acceptance_suite([case], _gen_good, split="probe", mode="trace_replay", tokenizer=_FakeTokenizer())
    assert len(prompts_seen) == 2
    assert rep.cases[0].passed is True
    assert rep.cases[0].recovered_after_failure is True

    # Negative control: recovery keywords inside Turn-2 <think> only must NOT pass recovery
    def _gen_hidden_in_think(prompt: str, max_tokens: int) -> str:
        if "TOOL OBSERVATION:" not in prompt:
            return "Thinking.\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>missing.txt</parameter>\n</function>\n</tool_call>"
        return "File does not exist so I should try fallback.\n</think>\n\nOK."

    rep_bad = run_acceptance_suite([case], _gen_hidden_in_think, split="probe", mode="trace_replay", tokenizer=_FakeTokenizer())
    assert rep_bad.cases[0].recovered_after_failure is False
    assert rep_bad.cases[0].passed is False
