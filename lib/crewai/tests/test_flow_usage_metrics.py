"""Tests for flow-level token usage aggregation

``flow.usage_metrics`` listens to ``LLMCallCompletedEvent`` for the duration
of ``kickoff_async`` so it covers every LLM call inside the flow — crew-led,
tool-led, AND bare ``LLM.call(...)`` from a flow method. We exercise the
aggregator end-to-end through the real event bus with fabricated events and
explicit contextvar control; no live LLM provider is required.
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable
from uuid import uuid4

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType
from crewai.flow.flow import Flow, start
from crewai.flow.flow_context import current_flow_id
from crewai.flow.runtime import _usage_dict_to_metrics
from crewai.types.usage_metrics import UsageMetrics


def _emit_llm_call(
    *,
    flow_id: str | None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_prompt_tokens: int = 0,
    reasoning_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> None:
    """Emit one fake ``LLMCallCompletedEvent`` with ``current_flow_id`` pinned
    to ``flow_id``.

    Runs in a freshly-copied context so the value the bus snapshots at emit
    time is exactly ``flow_id`` — independent of the calling thread's outer
    context. Mirrors how the real ``LLM.call`` emits events at runtime.
    """
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    for key, value in (
        ("cached_prompt_tokens", cached_prompt_tokens),
        ("reasoning_tokens", reasoning_tokens),
        ("cache_creation_tokens", cache_creation_tokens),
    ):
        if value:
            usage[key] = value
    event = LLMCallCompletedEvent(
        call_id=str(uuid4()),
        model="gpt-4o-mini",
        response="ok",
        call_type=LLMCallType.LLM_CALL,
        usage=usage,
    )

    ctx = contextvars.copy_context()

    def _emit() -> None:
        current_flow_id.set(flow_id)
        future = crewai_event_bus.emit(object(), event)
        if future is not None:
            future.result(timeout=5.0)

    ctx.run(_emit)


class _ScriptedFlow(Flow):
    """A Flow whose ``@start`` delegates to a per-instance ``_script`` closure.

    Each test attaches a script with ``flow._script = lambda f: ...`` so we
    don't redefine a Flow subclass for every scenario.
    """

    @start()
    def run(self) -> None:
        script: Callable[[Flow], None] = getattr(self, "_script", lambda _f: None)
        script(self)


def _run(script: Callable[[Flow], None] = lambda _f: None) -> Flow:
    """Build a ``_ScriptedFlow``, attach ``script``, kickoff. Returns the flow."""
    flow = _ScriptedFlow()
    flow._script = script
    flow.kickoff()
    return flow


class TestUsageDictToMetrics:
    """Unit tests for the dict-to-UsageMetrics normalizer."""

    @pytest.mark.parametrize(
        "usage, expected",
        [
            (None, None),
            ({}, None),
            (
                {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                UsageMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    successful_requests=1,
                ),
            ),
            # total_tokens missing → derived from prompt + completion
            (
                {"prompt_tokens": 4, "completion_tokens": 6},
                UsageMetrics(
                    prompt_tokens=4,
                    completion_tokens=6,
                    total_tokens=10,
                    successful_requests=1,
                ),
            ),
            # Extended provider-specific keys flow through normalization
            (
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 80,
                    "total_tokens": 180,
                    "cached_prompt_tokens": 40,
                    "reasoning_tokens": 25,
                    "cache_creation_tokens": 10,
                },
                UsageMetrics(
                    prompt_tokens=100,
                    completion_tokens=80,
                    total_tokens=180,
                    cached_prompt_tokens=40,
                    reasoning_tokens=25,
                    cache_creation_tokens=10,
                    successful_requests=1,
                ),
            ),
            # Garbage / non-int values coerce to 0 instead of crashing
            (
                {"prompt_tokens": "n/a", "completion_tokens": None, "total_tokens": 7},
                UsageMetrics(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=7,
                    successful_requests=1,
                ),
            ),
        ],
        ids=["none", "empty", "all_keys", "no_total", "extended_keys", "garbage"],
    )
    def test_normalization(
        self, usage: dict[str, Any] | None, expected: UsageMetrics | None
    ) -> None:
        assert _usage_dict_to_metrics(usage) == expected


class TestFlowUsageAggregation:
    """End-to-end tests driving the listener through the real event bus."""

    def test_sums_every_llm_call_in_the_flow(self) -> None:
        """Multiple LLM calls — including bare ``LLM.call(...)`` made outside
        any crew — accumulate; ``successful_requests`` tracks the call count."""

        def script(flow: Flow) -> None:
            _emit_llm_call(flow_id=flow._flow_match_id, prompt_tokens=300, completion_tokens=300)
            _emit_llm_call(flow_id=flow._flow_match_id, prompt_tokens=200, completion_tokens=100)
            _emit_llm_call(flow_id=flow._flow_match_id, prompt_tokens=20, completion_tokens=20)

        flow = _run(script)

        assert flow.usage_metrics.total_tokens == 940
        assert flow.usage_metrics.prompt_tokens == 520
        assert flow.usage_metrics.completion_tokens == 420
        assert flow.usage_metrics.successful_requests == 3

    def test_returns_zero_when_no_calls_happen(self) -> None:
        flow = _run()
        assert flow.usage_metrics == UsageMetrics()

    def test_ignores_events_from_other_flows(self) -> None:
        """Concurrent flow runs share the singleton bus, so the listener must
        scope itself to its own flow via the contextvar match."""

        def script(flow: Flow) -> None:
            _emit_llm_call(flow_id=flow._flow_match_id, prompt_tokens=50, completion_tokens=50)
            _emit_llm_call(flow_id="some-other-flow", prompt_tokens=49_000, completion_tokens=50_999)

        flow = _run(script)

        assert flow.usage_metrics.total_tokens == 100
        assert flow.usage_metrics.successful_requests == 1

    def test_resets_between_kickoffs(self) -> None:
        flow = _ScriptedFlow()
        flow._script = lambda f: _emit_llm_call(
            flow_id=f._flow_match_id, prompt_tokens=250, completion_tokens=250
        )

        flow.kickoff()
        flow.kickoff()

        assert flow.usage_metrics.total_tokens == 500
        assert flow.usage_metrics.successful_requests == 1

    def test_snapshot_is_immutable(self) -> None:
        flow = _run(
            lambda f: _emit_llm_call(
                flow_id=f._flow_match_id, prompt_tokens=50, completion_tokens=50
            )
        )

        snapshot = flow.usage_metrics
        snapshot.total_tokens = 999_999

        assert flow.usage_metrics.total_tokens == 100

    def test_handler_is_unregistered_after_kickoff(self) -> None:
        """Long-lived workers (Celery, devkit) must not leak one handler per
        kickoff on the singleton bus."""

        def handler_count() -> int:
            return len(
                crewai_event_bus._sync_handlers.get(LLMCallCompletedEvent, frozenset())
            )

        before = handler_count()

        flow = _ScriptedFlow()
        flow._script = lambda f: _emit_llm_call(
            flow_id=f._flow_match_id, prompt_tokens=5, completion_tokens=5
        )
        for _ in range(3):
            flow.kickoff()

        assert handler_count() == before
