"""Per-execution monotonic budget shared by concurrent research tasks.

The service still owns the outer cancellation deadline. This layer prevents
new work near that deadline; it neither retries providers nor releases costs.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import asyncio
import math
import time


class ResearchDeadlineExceeded(TimeoutError):
    failure_kind = 'research_deadline_expired'

    def __init__(self, stage, remaining_seconds, *, failure_kind="research_deadline_expired"):
        self.failure_kind = failure_kind
        self.stage = stage
        self.remaining_seconds = max(0.0, remaining_seconds)
        super().__init__(f'Research time budget cannot safely start or finish {stage}')


class ExecutionBudget:
    def __init__(self, seconds, *, clock=time.monotonic):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Execution budget must be finite and positive')
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + seconds
        self.events = []
        self.stage_spans = []
        self.provider_spans = []
        self.stage = "research"

    def remaining(self):
        return max(0.0, self.deadline - self.clock())

    def require(self, stage, *, minimum_seconds=0.01):
        remaining = self.remaining()
        if remaining < minimum_seconds:
            self.events.append({'stage':stage, 'event':'not_started', 'remaining_seconds':round(remaining, 3)})
            raise ResearchDeadlineExceeded(stage, remaining, failure_kind="insufficient_time_for_stage" if remaining > 0 else "research_deadline_expired")
        return remaining

    def set_stage(self, stage, role=None):
        elapsed = self.clock() - self.started
        if self.stage_spans:
            self.stage_spans[-1]["duration_seconds"] = round(elapsed - self.stage_spans[-1]["started_offset_seconds"], 3)
        self.stage = stage
        self.stage_spans.append({"stage":stage, "role":role, "started_offset_seconds":elapsed, "duration_seconds":None})

    def record_attempt(self, attempt):
        elapsed = self.clock() - self.started
        self.provider_spans.append({
            "call_id":attempt.call_id, "attempt_index":attempt.attempt_index,
            "provider":attempt.provider, "model":attempt.model, "role":attempt.role,
            "stage":self.stage, "ended_offset_seconds":round(elapsed, 3),
            "approximate_started_offset_seconds":round(max(0, elapsed-attempt.latency_ms/1000), 3),
            "latency_ms":attempt.latency_ms, "admission_wait_ms":attempt.admission_wait_ms,
            "success":attempt.success, "skipped":attempt.skipped, "error_type":attempt.error_type,
            "usage_known":attempt.input_tokens is not None and attempt.output_tokens is not None,
        })

    def snapshot(self):
        spans = [dict(span) for span in self.stage_spans]
        if spans:
            spans[-1]["duration_seconds"] = round(self.clock()-self.started-spans[-1]["started_offset_seconds"], 3)
        return {'version':'execution-budget-v1', 'elapsed_seconds':round(self.clock()-self.started, 3),
                'remaining_seconds':round(self.remaining(), 3), 'events':list(self.events),
                'stage_spans':spans, 'provider_spans':list(self.provider_spans)}


_active = ContextVar('scientific_research_budget', default=None)


def current_budget():
    return _active.get()


@contextmanager
def use_budget(budget):
    token = _active.set(budget)
    try:
        yield budget
    finally:
        _active.reset(token)


def require_budget(stage, *, minimum_seconds=0.01):
    budget = current_budget()
    return budget.require(stage, minimum_seconds=minimum_seconds) if budget else None


def capped_deadline(local_deadline, *, clock=time.monotonic):
    budget = current_budget()
    if budget is None:
        return local_deadline
    return min(local_deadline, clock() + budget.require('provider call'))


async def run_with_budget(stage, operation_factory, *, cap_seconds, minimum_seconds=0.01):
    """Create the coroutine only after admission; cancellation always propagates."""
    budget = current_budget()
    remaining = require_budget(stage, minimum_seconds=minimum_seconds)
    timeout = min(cap_seconds, remaining) if remaining is not None else cap_seconds
    started = budget.clock() if budget else None
    try:
        return await asyncio.wait_for(operation_factory(), timeout=timeout)
    except TimeoutError as error:
        # An inner provider timeout with global budget left remains a provider
        # failure. Only exhaustion of this stage allowance is a stage deadline.
        if budget and budget.clock() - started >= timeout:
            raise ResearchDeadlineExceeded(stage, budget.remaining(), failure_kind="stage_deadline_expired" if budget.remaining() > 0 else "research_deadline_expired") from error
        raise
    finally:
        if budget:
            budget.events.append({'stage':stage, 'event':'finished', 'duration_seconds':round(budget.clock()-started, 3),
                                  'remaining_seconds':round(budget.remaining(), 3)})
