"""Core data types shared by every layer (PRD section 14)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

QClass = Literal["L", "A", "P"]
AnswerType = Literal["count", "amount", "ticket_id", "customer_ids"]
Role = Literal["planner", "mapper", "aggregator", "direct"]


class Ticket(BaseModel):
    ticket_id: str
    ts: str
    customer_id: str
    product: str
    category: Literal["billing", "shipping", "defect", "account", "other"]
    refund_amount: Optional[float] = None
    sentiment: Literal["positive", "neutral", "negative"]
    text: str

    def line(self) -> str:
        """How the ticket appears in the log: structured header, free text. Labels are never shown."""
        return f"{self.ticket_id} | {self.ts} | {self.customer_id} | {self.product} | {self.text}"


class Question(BaseModel):
    question_id: str
    cls: QClass
    size: str
    text: str
    answer_type: AnswerType
    truth: Any
    seed: int
    template: str
    params: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    provider: str
    device: str
    model: str
    roles: list[Role]
    context_window: int
    price_in: float = 0.0
    price_out: float = 0.0
    rpm: Optional[int] = None
    tpm: Optional[int] = None
    rpd: Optional[int] = None
    tpd: Optional[int] = None
    max_concurrency: int = 8
    limit_safety: float = 1.0  # admit only this share of the published per-minute limits
    max_call_tokens: Optional[int] = None  # hard per-call cap (Groq free plan: 7000)
    reasoning_effort: Optional[str] = None
    in_main_pool: bool = True
    # latency priors used before profiling exists (vendor claims or measurements, labelled)
    ttft_ms: float = 300.0
    prefill_tps: float = 5000.0
    tps_out: float = 500.0
    watts: Optional[float] = None
    sim: dict[str, Any] = Field(default_factory=dict)


class Profile(BaseModel):
    candidate_id: str
    ttft_p50: float
    ttft_p95: float
    ttft_base: float = 0.0
    tps_out: float
    prefill_tps: float
    error_rate: float
    n_calls: int
    concurrency: int = 1
    measured_at: str = ""
    source: str = "sim"


class CallSpec(BaseModel):
    role: Role
    tokens_in_est: int
    tokens_out_est: int
    on_critical_path: bool
    features: dict[str, float] = Field(default_factory=dict)


class Decision(BaseModel):
    candidate_id: str
    q_hat: float
    l_hat: float
    c_hat: float
    propensity: float
    reason: str
    explored: bool = False
    flagged: bool = False


class Span(BaseModel):
    run_id: str
    question_id: str
    span_id: str
    parent_id: Optional[str] = None
    role: str
    candidate_id: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int = 0
    ttft_ms: float = 0.0
    queue_ms: float = 0.0
    total_ms: float = 0.0
    start_ms: float = 0.0
    end_ms: float = 0.0
    cost: float = 0.0
    cache_hit: bool = False
    retries: int = 0
    propensity: Optional[float] = None
    q_hat: Optional[float] = None
    l_hat: Optional[float] = None
    c_hat: Optional[float] = None
    explored: bool = False
    request_id: Optional[str] = None  # hash of the messages, so an offline replay can follow live routing
    ok: bool = True
    correct: Optional[bool] = None
    note: str = ""
    features: dict[str, float] = Field(default_factory=dict)


class RunRecord(BaseModel):
    run_id: str
    pipeline: str
    question_id: str
    cls: QClass
    size: str
    answer: Any = None
    truth: Any = None
    correct: bool = False
    malformed: bool = False
    truncated: bool = False
    cost: float = 0.0
    latency_ms: float = 0.0
    wall_ms: float = 0.0
    plan: Optional[dict] = None
    notes: list[str] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)


class Verdict(BaseModel):
    cls: str
    size: str
    n: int
    acc_a: float
    acc_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adj: Optional[float] = None
    margin: float
    outcome: Literal["non-inferior", "superior", "inferior", "inconclusive"]
    why: str
