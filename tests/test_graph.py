import pytest
from fakes import FakeLlm

from crucible.errors import CrucibleError
from crucible.graph import buildGraph, routeAfterJudge
from crucible.schemas import Claim, DebateState, Evidence, Verdict

Threshold = 0.75


def runGraph(Llm, MaxRounds: int = 3) -> DebateState:
    Graph = buildGraph(Llm, Threshold)
    StartState = DebateState(claim=Claim(text="A test claim"), max_rounds=MaxRounds)
    Result = Graph.invoke(StartState)
    # invoke() returns a plain dict, so we turn it back into a checked DebateState.
    return DebateState.model_validate(Result)


def getAgentOrder(FinalState: DebateState) -> list[str]:
    Order = []
    for Entry in FinalState.trace:
        Order.append(Entry.agent)
    return Order


def testHighConfidenceFinishesInOneRound():
    Llm = FakeLlm(Confidences=[0.90])

    FinalState = runGraph(Llm)

    assert getAgentOrder(FinalState) == [
        "Proposer",
        "Skeptic",
        "EvidenceGatherer",
        "Judge",
    ]
    assert FinalState.round_number == 1
    assert FinalState.verdict.confidence == 0.90
    assert FinalState.verdict.label == "MIXED"


def testLowConfidenceLoopsBackToSkepticNotProposer():
    Llm = FakeLlm(Confidences=[0.50, 0.90])

    FinalState = runGraph(Llm)

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
    assert FinalState.round_number == 2
    assert FinalState.verdict.confidence == 0.90
    # Proposer ran once and Skeptic ran twice, so 3 text calls and 2 judge calls.
    assert Llm.TextCallCount == 3
    assert Llm.JudgeRunner.CallCount == 2


def testEvidenceAccumulatesAcrossRounds():
    Llm = FakeLlm(Confidences=[0.50, 0.90])

    FinalState = runGraph(Llm)

    assert len(FinalState.evidence) == 4
    for Item in FinalState.evidence:
        assert isinstance(Item, Evidence)
    RoundNumbers = []
    for Item in FinalState.evidence:
        RoundNumbers.append(Item.round_number)
    assert RoundNumbers == [1, 1, 2, 2]


def testMaxRoundsStopsTheLoop():
    Llm = FakeLlm(Confidences=[0.20])

    FinalState = runGraph(Llm)

    assert FinalState.round_number == 3
    assert Llm.JudgeRunner.CallCount == 3
    assert FinalState.verdict.confidence == 0.20


def testLlmFailureIsNotSwallowed():
    Llm = FakeLlm(Confidences=[0.90], ErrorToRaise=RuntimeError("network down"))

    with pytest.raises(CrucibleError) as Caught:
        runGraph(Llm)

    assert "Proposer" in str(Caught.value)
    assert "network down" in str(Caught.value)


def testEmptyLlmAnswerIsNotSwallowed():
    Llm = FakeLlm(Confidences=[0.90], TextReply="   ")

    with pytest.raises(CrucibleError) as Caught:
        runGraph(Llm)

    assert "empty" in str(Caught.value)


def makeJudgedState(Confidence: float, RoundNumber: int) -> DebateState:
    Result = Verdict(label="MIXED", confidence=Confidence, reasoning="Because.")
    return DebateState(
        claim=Claim(text="A test claim"),
        verdict=Result,
        round_number=RoundNumber,
        max_rounds=3,
    )


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
    State = makeJudgedState(Confidence, RoundNumber)
    assert routeAfterJudge(State, Threshold) == Expected


def testRouteWithoutAVerdictRaisesAClearError():
    State = DebateState(claim=Claim(text="A test claim"))
    with pytest.raises(CrucibleError):
        routeAfterJudge(State, Threshold)
