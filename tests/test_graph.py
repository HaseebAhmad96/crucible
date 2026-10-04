import pytest
from fakes import FakeLlm

from crucible.errors import CrucibleError
from crucible.graph import buildGraph, routeAfterJudge
from crucible.state import makeInitialState

Threshold = 0.75


def getAgentOrder(FinalState) -> list[str]:
    return [Entry["agent"] for Entry in FinalState["trace"]]


def testHighConfidenceFinishesInOneRound():
    Llm = FakeLlm(Confidences=[0.90])
    Graph = buildGraph(Llm, Threshold)

    FinalState = Graph.invoke(makeInitialState("A claim", 3))

    assert getAgentOrder(FinalState) == [
        "Proposer",
        "Skeptic",
        "EvidenceGatherer",
        "Judge",
    ]
    assert FinalState["round_number"] == 1
    assert FinalState["confidence"] == 0.90
    assert FinalState["verdict"] == "MIXED"


def testLowConfidenceLoopsBackToSkepticNotProposer():
    Llm = FakeLlm(Confidences=[0.50, 0.90])
    Graph = buildGraph(Llm, Threshold)

    FinalState = Graph.invoke(makeInitialState("A claim", 3))

    assert getAgentOrder(FinalState) == [
        "Proposer",
        "Skeptic",
        "EvidenceGatherer",
        "Judge",
        "Loop",
        "Skeptic",
        "EvidenceGatherer",
        "Judge",
    ]
    assert FinalState["round_number"] == 2
    assert FinalState["confidence"] == 0.90
    # Proposer ran once and Skeptic ran twice, so 3 text calls and 2 judge calls.
    assert Llm.TextCallCount == 3
    assert Llm.JudgeRunner.CallCount == 2


def testEvidenceAccumulatesAcrossRounds():
    Llm = FakeLlm(Confidences=[0.50, 0.90])
    Graph = buildGraph(Llm, Threshold)

    FinalState = Graph.invoke(makeInitialState("A claim", 3))

    assert len(FinalState["evidence"]) == 4


def testMaxRoundsStopsTheLoop():
    Llm = FakeLlm(Confidences=[0.20])
    Graph = buildGraph(Llm, Threshold)

    FinalState = Graph.invoke(makeInitialState("A claim", 3))

    assert FinalState["round_number"] == 3
    assert Llm.JudgeRunner.CallCount == 3
    assert FinalState["confidence"] == 0.20


def testLlmFailureIsNotSwallowed():
    Llm = FakeLlm(Confidences=[0.90], ErrorToRaise=RuntimeError("network down"))
    Graph = buildGraph(Llm, Threshold)

    with pytest.raises(CrucibleError) as Caught:
        Graph.invoke(makeInitialState("A claim", 3))

    assert "Proposer" in str(Caught.value)
    assert "network down" in str(Caught.value)


def testEmptyLlmAnswerIsNotSwallowed():
    Llm = FakeLlm(Confidences=[0.90], TextReply="   ")
    Graph = buildGraph(Llm, Threshold)

    with pytest.raises(CrucibleError) as Caught:
        Graph.invoke(makeInitialState("A claim", 3))

    assert "empty" in str(Caught.value)


@pytest.mark.parametrize(
    "Confidence, RoundNumber, Expected",
    [
        (0.90, 1, "finish"),
        (0.75, 1, "finish"),
        (0.50, 1, "loop"),
        (0.50, 2, "loop"),
        (0.50, 3, "finish"),
    ],
)
def testRouteAfterJudge(Confidence, RoundNumber, Expected):
    State = {"confidence": Confidence, "round_number": RoundNumber, "max_rounds": 3}
    assert routeAfterJudge(State, Threshold) == Expected
