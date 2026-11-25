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
            "transmission":(attempt.diagnostics or {}).get("transmission", "not_recorded"),
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


class StageAllowance:
    def __init__(self, stage, deadline, clock):
        self.stage, self.deadline, self.clock = stage, deadline, clock

    def remaining(self):
        return max(0.0, self.deadline-self.clock())


_stage = ContextVar('scientific_stage_allowance', default=None)


def current_stage_budget():
    return _stage.get()


def stage_admission_refused():
    stage = current_stage_budget()
    if stage is not None:
        raise ResearchDeadlineExceeded(stage.stage, stage.remaining(), failure_kind="stage_admission_refused")


def provider_attempt_fits(timeout_seconds):
    stage = current_stage_budget()
    # Leave a small cancellation/finalization margin; do not shrink a provider's
    # allowed window to conceal a routed wait that cannot fit the stage.
    return stage is None or stage.remaining() >= timeout_seconds + 1.0


def capped_deadline(local_deadline, *, clock=time.monotonic):
    budget = current_budget()
    remaining = budget.require('provider call') if budget else None
    stage = current_stage_budget()
    if stage is not None:
        remaining = min(remaining, stage.remaining()) if remaining is not None else stage.remaining()
        if remaining <= 0:
            stage_admission_refused()
    return min(local_deadline, clock()+remaining) if remaining is not None else local_deadline


async def run_with_budget(stage, operation_factory, *, cap_seconds, minimum_seconds=0.01, reserve_after_seconds=0.0):
    """Admit before creating a coroutine and propagate the innermost deadline.

    Mandatory downstream work is reserved from the global allowance. Cancellation
    is never caught or converted into fallback or an unverifiable partial answer.
    """
    budget = current_budget()
    remaining = require_budget(stage, minimum_seconds=minimum_seconds+reserve_after_seconds)
    clock = budget.clock if budget else time.monotonic
    started = clock()
    allowance = min(cap_seconds, remaining-reserve_after_seconds) if remaining is not None else cap_seconds
    parent = current_stage_budget()
    if parent is not None:
        allowance = min(allowance, parent.remaining())
    child = StageAllowance(stage, started+allowance, clock)
    token = _stage.set(child)
    try:
        result = await asyncio.wait_for(operation_factory(), timeout=max(0.001,allowance))
        if child.remaining() <= 0:
            raise ResearchDeadlineExceeded(stage, budget.remaining() if budget else 0, failure_kind="stage_deadline_expired")
        return result
    except ResearchDeadlineExceeded:
        raise
    except TimeoutError as error:
        if child.remaining() <= 0:
            raise ResearchDeadlineExceeded(stage, budget.remaining() if budget else 0,
                failure_kind="stage_deadline_expired" if budget is None or budget.remaining() > 0 else "research_deadline_expired") from error
        raise
    finally:
        _stage.reset(token)
        if budget:
            budget.events.append({'stage':stage,'event':'finished','duration_seconds':round(clock()-started,3),
                'remaining_seconds':round(budget.remaining(),3),'reserved_downstream_seconds':reserve_after_seconds})
