"""Builds the LangGraph workflow. In v0.1 the route through the graph is fixed."""

from functools import partial

from langgraph.graph import END, START, StateGraph

from crucible.nodes import (
    evidenceGathererNode,
    judgeNode,
    nextRoundNode,
    proposerNode,
    skepticNode,
)
from crucible.state import CrucibleState


def routeAfterJudge(State: CrucibleState, Threshold: float) -> str:
    """Decide what happens after the Judge. Returns a label, not a node name."""
    if State["confidence"] >= Threshold:
        return "finish"
    if State["round_number"] >= State["max_rounds"]:
        return "finish"
    return "loop"


def buildGraph(Llm, Threshold: float):
    Builder = StateGraph(CrucibleState)

    # partial(...) pre-fills the Llm argument, because LangGraph calls a node
    # with only the state.
    Builder.add_node("proposer", partial(proposerNode, Llm=Llm))
    Builder.add_node("skeptic", partial(skepticNode, Llm=Llm))
    Builder.add_node("evidence_gatherer", evidenceGathererNode)
    Builder.add_node("judge", partial(judgeNode, Llm=Llm))
    Builder.add_node("next_round", nextRoundNode)

    Builder.add_edge(START, "proposer")
    Builder.add_edge("proposer", "skeptic")
    Builder.add_edge("skeptic", "evidence_gatherer")
    Builder.add_edge("evidence_gatherer", "judge")

    # The conditional edge: after "judge", call the router and follow its label.
    Builder.add_conditional_edges(
        "judge",
        partial(routeAfterJudge, Threshold=Threshold),
        {"loop": "next_round", "finish": END},
    )
    Builder.add_edge("next_round", "skeptic")

    return Builder.compile()
