"""The shared state that travels between the graph nodes."""

import operator
from typing import Annotated, TypedDict


class TraceEntry(TypedDict):
    round_number: int
    agent: str
    text: str


class CrucibleState(TypedDict):
    claim: str
    proposal: str
    criticism: str
    # Fields wrapped in Annotated[..., operator.add] ACCUMULATE: when a node returns
    # a list, LangGraph appends it to the existing list. Every other field is
    # simply overwritten by whatever a node returns.
    evidence: Annotated[list[str], operator.add]
    verdict: str
    verdict_reason: str
    confidence: float
    round_number: int
    max_rounds: int
    trace: Annotated[list[TraceEntry], operator.add]


def makeInitialState(Claim: str, MaxRounds: int) -> CrucibleState:
    return {
        "claim": Claim,
        "proposal": "",
        "criticism": "",
        "evidence": [],
        "verdict": "",
        "verdict_reason": "",
        "confidence": 0.0,
        "round_number": 1,
        "max_rounds": MaxRounds,
        "trace": [],
    }


def makeTraceEntry(RoundNumber: int, Agent: str, Text: str) -> TraceEntry:
    return {"round_number": RoundNumber, "agent": Agent, "text": Text}
