"""Offline operational evaluation (work-order §9).

Measures coding correctness, tool selection, tool-call syntax and argument
semantics, responses to tool observations, recovery after a failed operation,
multi-turn continuation, correct finalization, general reasoning preservation,
long-context stability, and token efficiency.

Clearly distinguishes standalone generation, offline trace replay, and real
harness-assisted execution. The model may underperform its BF16 teacher on
isolated language metrics if it achieves a materially smaller footprint while
retaining enough semantic competence for the harness — but syntax validation
and retries cannot compensate for persistent misunderstanding of tasks.

Simulated execution success is never represented as real task completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    """One offline evaluation case."""

    case_id: str
    category: str  # "coding" | "tool_selection" | "tool_syntax" | "recovery" | ...
    prompt: str
    expected_tool: str | None = None
    expected_args: dict | None = None
    tool_observation: str | None = None  # injected observation for response testing
    max_tokens: int = 4096
    test_code: str | None = None
    expected_substrings: tuple[str, ...] | list[str] | None = None
    enable_thinking: bool = False
    split: str = "probe"


@dataclass(frozen=True)
class EvalResult:
    """Result of one evaluation case."""

    case_id: str
    category: str
    mode: str  # "standalone" | "trace_replay" | "harness_assisted"
    passed: bool
    syntax_valid: bool
    tool_selected: str | None
    args_match: bool | None
    recovered_after_failure: bool | None
    finalized_correctly: bool | None
    tokens_used: int
    notes: str = ""
    thinking_valid: bool | None = None
    code_passed: bool | None = None


@dataclass(frozen=True)
class EvaluationReport:
    """Offline evaluation summary (work-order §9)."""

    mode: str
    cases: list[EvalResult] = field(default_factory=list)
    coding_correctness: float | None = None
    tool_selection_accuracy: float | None = None
    tool_syntax_validity: float | None = None
    recovery_rate: float | None = None
    finalization_rate: float | None = None
    reasoning_preserved: bool | None = None
    long_context_stable: bool | None = None
    token_efficiency: float | None = None
    simulated_as_real: bool = False  # must stay False; enforced by construction
    notes: str = ""


SEALED_SPLITS = frozenset({"heldout", "eval_firewall"})


def _check_thinking_prompt_contract(prompt: str, *, enable_thinking: bool) -> bool:
    """Verify Qwen3.5/Ornith prompt-<think> contract before generation."""
    if not enable_thinking:
        # Thinking-disabled prompts must not leave an unclosed <think> tag.
        return prompt.count("<think>") <= prompt.count("</think>")
    # Thinking-enabled rendered prompts must end with an open <think>.
    if "<|im_start|>" in prompt or "<think>" in prompt or "</think>" in prompt:
        return prompt.count("<think>") > prompt.count("</think>")
    return True


def format_thinking_prompt(
    prompt: str,
    *,
    tokenizer: Any = None,
    enable_thinking: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Render ``prompt`` honoring the Ornith thinking-on prompt-<think> contract.

    If ``prompt`` already contains chat template markers or ``<think>`` tags,
    validate and return it unchanged rather than double-wrapping.
    """
    from .errors import QualityGateError

    if not _check_thinking_prompt_contract(prompt, enable_thinking=enable_thinking):
        raise QualityGateError(
            f"Prompt violates thinking={'on' if enable_thinking else 'off'} <think> contract"
        )
    if "<|im_start|>" in prompt or "<think>" in prompt:
        return prompt
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if tools is not None:
            kwargs["tools"] = tools
        return str(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs))
    if enable_thinking:
        return f"{prompt.rstrip()}\n<think>\n"
    return prompt


def extract_thinking_and_response(
    output: str,
    *,
    prompt: str = "",
    require_thinking: bool = False,
) -> tuple[bool, str | None, str]:
    """Separate `<think>` reasoning from the visible response.

    Honors the prompt-`<think>` contract: when ``prompt`` opens `<think>\n`
    (as in ``apply_chat_template(..., enable_thinking=True)``), valid model
    output emits reasoning up to ``</think>`` followed by visible answer text.
    """
    if require_thinking and not _check_thinking_prompt_contract(
        prompt, enable_thinking=True
    ):
        return False, None, ""

    prompt_opened_think = prompt.count("<think>") > prompt.count("</think>")
    if prompt_opened_think:
        if "</think>" not in output:
            return (False, None, "") if require_thinking else (True, None, output)
        before, after = output.split("</think>", 1)
        if "<think>" in before or "<think>" in after or "</think>" in after:
            return False, None, ""
        reasoning = before.strip()
        visible = after.strip()
        if require_thinking and (not reasoning or not visible):
            return False, reasoning or None, visible
        return True, reasoning, visible

    # Prompt did not leave an unclosed <think>.
    if "<think>" in output or "</think>" in output:
        open_idx = output.find("<think>")
        close_idx = output.find("</think>")
        if open_idx == -1 or close_idx == -1 or close_idx < open_idx:
            return False, None, ""
        reasoning = output[open_idx + len("<think>") : close_idx].strip()
        prefix = output[:open_idx].strip()
        suffix = output[close_idx + len("</think>") :].strip()
        if "<think>" in suffix or "</think>" in suffix:
            return False, None, ""
        visible = f"{prefix}\n{suffix}".strip() if (prefix and suffix) else (suffix or prefix)
        if require_thinking and (not reasoning or not visible):
            return False, reasoning or None, visible
        return True, reasoning, visible

    if require_thinking:
        return False, None, output.strip()
    return True, None, output


def parse_ornith_tool_calls(output: str, *, prompt: str = "") -> tuple[bool, list[dict[str, Any]]]:
    """Strictly parse Ornith/Qwen3.5 XML ``<tool_call><function=...>`` blocks.

    Returns ``(syntax_valid, calls)`` where each item has ``name`` and
    ``arguments`` (raw string values mapped by parameter name). Rejects
    unclosed tags, non-XML prose, duplicate parameters, and stray non-whitespace
    content inside ``<tool_call>`` or ``<function=...>`` blocks. Never extracts
    tool calls from inside ``<think>...</think>``.
    """
    import re

    thinking_ok, _reasoning, visible = extract_thinking_and_response(
        output, prompt=prompt, require_thinking=False
    )
    if not thinking_ok:
        return False, []

    if "<tool_call>" not in visible and "</tool_call>" not in visible:
        return False, []
    if visible.count("<tool_call>") != visible.count("</tool_call>"):
        return False, []

    block_matches = list(re.finditer(r"<tool_call>([\s\S]*?)</tool_call>", visible))
    if not block_matches or len(block_matches) != visible.count("<tool_call>"):
        return False, []

    # Reject nested or dangling tags outside matched blocks.
    outside = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", visible)
    if "<tool_call>" in outside or "</tool_call>" in outside:
        return False, []

    calls: list[dict[str, Any]] = []
    fn_pattern = re.compile(r"^\s*<function=([A-Za-z_][A-Za-z0-9_.\-]*)>([\s\S]*?)</function>\s*$")
    param_pattern = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_.\-]*)>([\s\S]*?)</parameter>")

    for match in block_matches:
        body = match.group(1)
        if "<tool_call>" in body or body.count("<function=") != 1 or body.count("</function>") != 1:
            return False, []
        fn_match = fn_pattern.match(body)
        if not fn_match:
            return False, []
        fn_name = fn_match.group(1)
        fn_body = fn_match.group(2)

        params = list(param_pattern.finditer(fn_body))
        if fn_body.count("<parameter=") != len(params) or fn_body.count("</parameter>") != len(params):
            return False, []
        leftover = param_pattern.sub("", fn_body).strip()
        if leftover:
            return False, []

        args: dict[str, str] = {}
        for p in params:
            key = p.group(1)
            val = p.group(2)
            if key in args:
                return False, []
            # Strip leading/trailing newline added by multi-line XML formatting.
            if val.startswith("\n"):
                val = val[1:]
            if val.endswith("\n"):
                val = val[:-1]
            args[key] = val

        calls.append({"name": fn_name, "arguments": args})

    return bool(calls), calls


def _extract_python_code(response: str) -> str:
    """Extract Python code from fenced markdown blocks or raw Python text."""
    import re

    py_blocks = re.findall(r"```(?:python|py)\s*\n([\s\S]*?)```", response, flags=re.I)
    if py_blocks:
        return "\n\n".join(b.strip("\n") for b in py_blocks)
    generic_blocks = re.findall(r"```\s*\n([\s\S]*?)```", response)
    if generic_blocks:
        return "\n\n".join(b.strip("\n") for b in generic_blocks)
    return response.strip()


def _evaluate_coding_output(
    response: str,
    *,
    test_code: str | None = None,
    timeout_s: float = 5.0,
) -> tuple[bool, bool, str]:
    """Validate Python syntax and execute solution + assertions in an isolated subprocess."""
    import ast
    import subprocess
    import sys

    code = _extract_python_code(response)
    if not code.strip():
        return False, False, "empty Python solution"

    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, False, f"SyntaxError in coding output: {exc}"

    if test_code is not None:
        try:
            ast.parse(test_code)
        except SyntaxError as exc:
            return True, False, f"SyntaxError in test_code assertions: {exc}"
        script = f"{code}\n\n{test_code}\n"
    else:
        script = f"{code}\n"

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return True, False, f"coding execution timed out after {timeout_s:.1f}s"

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        tail = err.splitlines()[-1] if err else f"exit code {proc.returncode}"
        return True, False, f"coding assertion failed: {tail}"
    return True, True, "coding assertions passed"


def evaluate_case(
    case: EvalCase,
    generate_fn: Any,
    mode: str = "standalone",
) -> EvalResult:
    """Run one evaluation case through a generation function.

    ``generate_fn(prompt, max_tokens) -> str`` is the model's generation entry
    point. This evaluator parses the output for tool calls, checks syntax
    validity, and (in trace_replay mode) injects ``case.tool_observation`` and
    re-generates to test responses to tool observations.

    Does NOT execute the tool — that is real harness-assisted execution, which
    this offline evaluator cannot perform. Mode is recorded verbatim so the
    report never conflates offline replay with real execution.
    """
    prompt_contract_ok = _check_thinking_prompt_contract(
        case.prompt, enable_thinking=case.enable_thinking
    )
    effective_prompt = (
        format_thinking_prompt(case.prompt, enable_thinking=True)
        if (case.enable_thinking and prompt_contract_ok)
        else case.prompt
    )
    output = generate_fn(effective_prompt, case.max_tokens)
    thinking_ok, thinking_text, visible_output = extract_thinking_and_response(
        output,
        prompt=effective_prompt,
        require_thinking=case.enable_thinking,
    )
    if not prompt_contract_ok:
        thinking_ok = False

    tool_selected: str | None = None
    args_match: bool | None = None
    code_exec_ok = True

    if case.expected_tool is not None:
        xml_ok, calls = parse_ornith_tool_calls(output, prompt=effective_prompt)
        syntax_valid = bool(xml_ok)
        if xml_ok and calls:
            matched = next((c for c in calls if c["name"] == case.expected_tool), calls[0])
            tool_selected = matched["name"]
        if case.expected_args is not None:
            if xml_ok and tool_selected == case.expected_tool:
                args_match = _args_match(
                    output,
                    case.expected_args,
                    expected_tool=case.expected_tool,
                    prompt=effective_prompt,
                )
            else:
                args_match = False
    elif case.category == "tool_syntax":
        xml_ok, calls = parse_ornith_tool_calls(output, prompt=effective_prompt)
        syntax_valid = bool(xml_ok)
        tool_selected = calls[0]["name"] if (xml_ok and calls) else None
        if case.expected_args is not None:
            args_match = (
                _args_match(output, case.expected_args, prompt=effective_prompt)
                if xml_ok
                else False
            )
    elif case.category == "coding":
        syntax_valid, code_exec_ok, _code_msg = _evaluate_coding_output(
            visible_output,
            test_code=case.test_code,
        )
    else:
        syntax_valid = bool(thinking_ok and visible_output.strip())

    recovered = None
    finalized = None
    tokens_used = len(output.split())

    if mode == "trace_replay" and case.tool_observation is not None:
        # Inject the tool observation and test the model's response.
        followup = effective_prompt + "\n\nTOOL OBSERVATION: " + case.tool_observation + "\n\nContinue."
        followup_output = generate_fn(followup, case.max_tokens)
        _, _, followup_visible = extract_thinking_and_response(
            followup_output,
            prompt=followup,
            require_thinking=False,
        )
        recovered = _indicates_recovery(followup_visible)
        finalized = _indicates_finalization(followup_visible)
        tokens_used += len(followup_output.split())

    substrings_ok = (
        True
        if not case.expected_substrings
        else all(str(s) in visible_output for s in case.expected_substrings)
    )
    recovery_ok = (
        bool(recovered)
        if (mode == "trace_replay" and case.tool_observation is not None and case.category == "recovery")
        else True
    )
    if case.expected_tool is not None:
        passed = bool(
            thinking_ok
            and syntax_valid
            and tool_selected == case.expected_tool
            and (args_match is True or args_match is None)
            and substrings_ok
            and recovery_ok
        )
    elif case.category == "tool_syntax":
        passed = bool(
            thinking_ok
            and syntax_valid
            and tool_selected is not None
            and (args_match is True or args_match is None)
            and substrings_ok
        )
    elif case.category == "coding":
        passed = bool(thinking_ok and syntax_valid and code_exec_ok and substrings_ok)
    else:
        passed = bool(thinking_ok and syntax_valid and recovery_ok and substrings_ok)

    return EvalResult(
        case_id=case.case_id,
        category=case.category,
        mode=mode,
        passed=passed,
        syntax_valid=syntax_valid,
        tool_selected=tool_selected,
        args_match=args_match,
        recovered_after_failure=recovered,
        finalized_correctly=finalized,
        tokens_used=tokens_used,
    )


def _is_valid_tool_call_syntax(output: str, *, prompt: str = "") -> bool:
    """Strictly verify Ornith/Qwen3.5 XML ``<tool_call><function=...>`` syntax."""
    ok, calls = parse_ornith_tool_calls(output, prompt=prompt)
    return bool(ok and calls)


def _extract_tool_name(output: str, *, prompt: str = "") -> str | None:
    """Extract tool function name from parsed Ornith XML ``<tool_call>`` block."""
    ok, calls = parse_ornith_tool_calls(output, prompt=prompt)
    if not ok or not calls:
        return None
    return str(calls[0]["name"])


def _arg_value_matches(raw_val: str, expected: Any) -> bool:
    """Typed exact equality between XML parameter text and expected argument value."""
    import json
    import math

    stripped = raw_val.strip()
    if isinstance(expected, bool):
        low = stripped.lower()
        if low == "true":
            return expected is True
        if low == "false":
            return expected is False
        return False
    if isinstance(expected, int):
        try:
            return int(stripped) == expected
        except ValueError:
            return False
    if isinstance(expected, float):
        try:
            return math.isclose(float(stripped), expected, rel_tol=1e-6, abs_tol=1e-9)
        except ValueError:
            return False
    if isinstance(expected, (dict, list)):
        try:
            return json.loads(stripped) == expected
        except (ValueError, TypeError):
            return False
    if isinstance(expected, str):
        return raw_val == expected or stripped == expected.strip()
    return stripped == str(expected).strip()


def _args_match(
    output: str,
    expected: dict,
    *,
    expected_tool: str | None = None,
    prompt: str = "",
) -> bool:
    """Strictly match parsed ``<parameter=...>`` names and values against ``expected``."""
    ok, calls = parse_ornith_tool_calls(output, prompt=prompt)
    if not ok or not calls:
        return False
    call = (
        next((c for c in calls if c["name"] == expected_tool), calls[0])
        if expected_tool is not None
        else calls[0]
    )
    if expected_tool is not None and call["name"] != expected_tool:
        return False
    args = call["arguments"]
    for key, exp_val in expected.items():
        if key not in args:
            return False
        if not _arg_value_matches(args[key], exp_val):
            return False
    return True


def _indicates_recovery(output: str) -> bool:
    out = output.lower()
    return any(t in out for t in ("retry", "recover", "fallback", "instead", "corrected"))


def _indicates_finalization(output: str) -> bool:
    out = output.lower()
    return any(t in out for t in ("final", "answer", "conclusion", "result:", "done"))


def aggregate(results: list[EvalResult], mode: str) -> EvaluationReport:
    """Aggregate case results into an :class:`EvaluationReport`.

    Rates are fractions of cases where the relevant field is not None and True.
    ``simulated_as_real`` is forced False — this evaluator never claims offline
    replay is real harness execution.
    """
    total = len(results) or 1
    coding = [r for r in results if r.category == "coding"]
    tool_sel = [r for r in results if r.category == "tool_selection"]
    tool_syn = [r for r in results if r.category == "tool_syntax"]
    rec = [r for r in results if r.recovered_after_failure is not None]
    fin = [r for r in results if r.finalized_correctly is not None]

    return EvaluationReport(
        mode=mode,
        cases=list(results),
        coding_correctness=(sum(r.passed for r in coding) / len(coding)) if coding else None,
        tool_selection_accuracy=(
            sum(r.passed for r in tool_sel) / len(tool_sel) if tool_sel else None
        ),
        tool_syntax_validity=(
            sum(r.syntax_valid for r in tool_syn) / len(tool_syn) if tool_syn else None
        ),
        recovery_rate=(sum(r.recovered_after_failure for r in rec) / len(rec)) if rec else None,
        finalization_rate=(sum(r.finalized_correctly for r in fin) / len(fin)) if fin else None,
        reasoning_preserved=None,
        long_context_stable=None,
        token_efficiency=(sum(r.tokens_used for r in results) / total if results else None),
        simulated_as_real=False,
    )


def run_acceptance_suite(
    cases: Any,
    generate_fn: Any,
    *,
    split: str = "probe",
    mode: str = "standalone",
    allow_heldout: bool = False,
    tokenizer: Any = None,
    ce_measurements: dict[str, float] | None = None,
    max_ce_regression: float = 0.50,
    ref_logits: Any = None,
    reload_logits: Any = None,
    logits_tolerance: float = 1e-2,
    coding_timeout_s: float = 5.0,
) -> EvaluationReport:
    """Run the offline acceptance suite with strict XML, coding, and thinking checks.

    Seals ``heldout`` / ``eval_firewall`` splits unless ``allow_heldout=True`` is
    passed explicitly for final blinded verification.

    # Errors: raises ``QualityGateError`` if ``split`` (or any case's ``split``) is
    sealed while ``allow_heldout=False``, if ``cases`` is empty, or if supplied
    CE / logits / top-1 gates fail.
    """
    from dataclasses import replace
    from .errors import QualityGateError
    from .gates import gate_ce_quality, gate_logits_parity, gate_top1_parity

    norm_split = str(split).strip().lower()
    if norm_split in SEALED_SPLITS and not allow_heldout:
        raise QualityGateError(
            f"Split {split!r} is sealed against development/probe runs; pass allow_heldout=True only for final evaluation"
        )

    case_list = list(cases)
    if not case_list:
        raise QualityGateError("Acceptance suite requires at least one EvalCase")

    for c in case_list:
        c_split = str(getattr(c, "split", "probe")).strip().lower()
        if c_split in SEALED_SPLITS and not allow_heldout:
            raise QualityGateError(
                f"Case {c.case_id!r} belongs to sealed split {c.split!r}; refusing unsealed access"
            )

    if ce_measurements is not None:
        gate_ce_quality(
            ce_measurements["teacher_ce"],
            ce_measurements["student_ce"],
            max_regression=max_ce_regression,
        )
    if ref_logits is not None and reload_logits is not None:
        gate_logits_parity(ref_logits, reload_logits, tolerance=logits_tolerance)
        gate_top1_parity(ref_logits, reload_logits)

    results: list[EvalResult] = []
    for case in case_list:
        captured: dict[str, str] = {}

        def _capture_gen(prompt_str: str, max_toks: int) -> str:
            resp = generate_fn(prompt_str, max_toks)
            if "output" not in captured:
                captured["prompt"] = prompt_str
                captured["output"] = resp
            return resp

        prompt_contract_ok = _check_thinking_prompt_contract(
            case.prompt, enable_thinking=case.enable_thinking
        )
        effective_case = case
        if case.enable_thinking and prompt_contract_ok:
            prepared = format_thinking_prompt(
                case.prompt,
                tokenizer=tokenizer,
                enable_thinking=True,
            )
            effective_case = replace(case, prompt=prepared)

        base_res = evaluate_case(effective_case, _capture_gen, mode=mode)
        raw_out = captured.get("output", "")
        raw_prompt = captured.get("prompt", effective_case.prompt)

        thinking_valid: bool | None = None
        thinking_ok, _reasoning, visible = extract_thinking_and_response(
            raw_out,
            prompt=raw_prompt,
            require_thinking=case.enable_thinking,
        )
        if not prompt_contract_ok:
            thinking_ok = False
        if case.enable_thinking or "<think>" in raw_out or "</think>" in raw_out:
            thinking_valid = bool(thinking_ok)

        code_passed: bool | None = None
        syntax_valid = base_res.syntax_valid
        passed = base_res.passed
        note_parts = [base_res.notes] if base_res.notes else []

        if case.category == "coding":
            c_syn, c_ok, c_msg = _evaluate_coding_output(
                visible,
                test_code=case.test_code,
                timeout_s=coding_timeout_s,
            )
            syntax_valid = c_syn
            code_passed = c_ok
            sub_ok = (
                True
                if not case.expected_substrings
                else all(str(s) in visible for s in case.expected_substrings)
            )
            passed = bool(thinking_ok and c_syn and c_ok and sub_ok)
            note_parts.append(c_msg)

        if thinking_valid is False:
            passed = False
            note_parts.append("thinking-<think>-contract failed")

        results.append(
            replace(
                base_res,
                passed=passed,
                syntax_valid=syntax_valid,
                thinking_valid=thinking_valid,
                code_passed=code_passed,
                notes="; ".join(p for p in note_parts if p),
            )
        )

    report = aggregate(results, mode)
    thinking_checks = [r.thinking_valid for r in results if r.thinking_valid is not None]
    reasoning_preserved = all(thinking_checks) if thinking_checks else None
    if report.reasoning_preserved is None and reasoning_preserved is not None:
        report = replace(report, reasoning_preserved=reasoning_preserved)
    return report


__all__ = [
    "SEALED_SPLITS",
    "EvalCase",
    "EvalResult",
    "EvaluationReport",
    "format_thinking_prompt",
    "extract_thinking_and_response",
    "parse_ornith_tool_calls",
    "evaluate_case",
    "aggregate",
    "run_acceptance_suite",
]
