"""Builds the LangGraph workflow. In this version the route is still fixed.

The graph knows nothing about prompts or LLM calls. Each node is simply the
`run` method of an agent object, so the node "functions" are as thin as possible.
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from crucible.agents import EvidenceGatherer, Judge, Proposer, Skeptic
from crucible.errors import CrucibleError
from crucible.schemas import DebateState, TraceEntry


def routeAfterJudge(State: DebateState, Threshold: float) -> str:
    """Decide what happens after the Judge. Returns a label, not a node name."""
    if State.verdict is None:
        raise CrucibleError("The router ran before the Judge produced a verdict.")
    if State.verdict.confidence >= Threshold:
        return "finish"
    if State.round_number >= State.max_rounds:
        return "finish"
    return "loop"


def nextRoundNode(State: DebateState) -> dict:
    """Bookkeeping, not an agent: move the counter to the next round."""
    NewRound = State.round_number + 1
    Confidence = State.verdict.confidence if State.verdict is not None else 0.0
    Text = (
        f"Confidence {Confidence:.2f} is too low. "
        f"Looping back to the Skeptic (round {NewRound} of {State.max_rounds})."
    )
    # This entry belongs to the round that just ended, so it keeps the OLD number.
    Entry = TraceEntry(round_number=State.round_number, agent="Loop", text=Text)
    return {"round_number": NewRound, "trace": [Entry]}


def buildGraph(Llm, Threshold: float):
    ProposerAgent = Proposer(Llm)
    SkepticAgent = Skeptic(Llm)
    EvidenceAgent = EvidenceGatherer()
    JudgeAgent = Judge(Llm)

    Builder = StateGraph(DebateState)

    Builder.add_node("proposer", ProposerAgent.run)
    Builder.add_node("skeptic", SkepticAgent.run)
    Builder.add_node("evidence_gatherer", EvidenceAgent.run)
    Builder.add_node("judge", JudgeAgent.run)
    Builder.add_node("next_round", nextRoundNode)

    Builder.add_edge(START, "proposer")
    Builder.add_edge("proposer", "skeptic")
    Builder.add_edge("skeptic", "evidence_gatherer")
    Builder.add_edge("evidence_gatherer", "judge")

    # partial(...) pre-fills Threshold, because LangGraph calls the router with
    # only the state.
    Builder.add_conditional_edges(
        "judge",
        partial(routeAfterJudge, Threshold=Threshold),
        {"loop": "next_round", "finish": END},
    )
    Builder.add_edge("next_round", "skeptic")

    return Builder.compile()
