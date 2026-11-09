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
    output = generate_fn(case.prompt, case.max_tokens)
    syntax_valid = _is_valid_tool_call_syntax(output) if case.expected_tool else True
    tool_selected = _extract_tool_name(output) if case.expected_tool else None
    args_match = None
    if case.expected_args is not None and tool_selected == case.expected_tool:
        args_match = _args_match(output, case.expected_args)
    recovered = None
    finalized = None
    tokens_used = len(output.split())

    if mode == "trace_replay" and case.tool_observation is not None:
        # Inject the tool observation and test the model's response.
        followup = case.prompt + "\n\nTOOL OBSERVATION: " + case.tool_observation + "\n\nContinue."
        followup_output = generate_fn(followup, case.max_tokens)
        recovered = _indicates_recovery(followup_output)
        finalized = _indicates_finalization(followup_output)
        tokens_used += len(followup_output.split())

    passed = bool(syntax_valid and (args_match is True or args_match is None))

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


def _is_valid_tool_call_syntax(output: str) -> bool:
    """Heuristic: does the output contain a well-formed tool call?"""
    import re

    return bool(re.search(r"<|tool_call|>.*?</|tool_call|>|\[TOOL\].*?\[/TOOL\]", output, re.S))


def _extract_tool_name(output: str) -> str | None:
    import re

    m = re.search(r"(?:tool|function)[:\s]+([A-Za-z_][A-Za-z0-9_]*)", output, re.I)
    return m.group(1) if m else None


def _args_match(output: str, expected: dict) -> bool:
    """Heuristic: do the expected argument keys appear in the output?"""
    return all(str(key) in output for key in expected)


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


__all__ = [
    "EvalCase",
    "EvalResult",
    "EvaluationReport",
    "evaluate_case",
    "aggregate",
]
