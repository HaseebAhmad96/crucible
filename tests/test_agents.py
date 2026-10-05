import pytest
from fakes import FakeLlm

from crucible.agents import BaseAgent, EvidenceGatherer, Judge, Proposer, Skeptic
from crucible.errors import CrucibleError
from crucible.schemas import Claim, DebateState, Evidence, Verdict


def makeState(**Overrides) -> DebateState:
    return DebateState(claim=Claim(text="A test claim"), **Overrides)


def getLastUserPrompt(Messages) -> str:
    # Messages is [SystemMessage, HumanMessage]; the user prompt is the second one.
    return Messages[-1].content


def testBaseAgentCannotBeCreatedDirectly():
    with pytest.raises(TypeError):
        BaseAgent()


def testEveryAgentIsABaseAgent():
    for AgentClass in [Proposer, Skeptic, EvidenceGatherer, Judge]:
        assert issubclass(AgentClass, BaseAgent)


def testProposerReturnsProposalAndTrace():
    Llm = FakeLlm(Confidences=[0.9], TextReply="My proposal")

    Update = Proposer(Llm).run(makeState())

    assert set(Update) == {"proposal", "trace"}
    assert Update["proposal"] == "My proposal"
    assert Update["trace"][0].agent == "Proposer"
    assert Update["trace"][0].round_number == 1


def testSkepticInRoundOneHasNoJudgeConcern():
    Llm = FakeLlm(Confidences=[0.9], TextReply="My criticism")

    Update = Skeptic(Llm).run(makeState(proposal="P"))

    assert Update["criticism"] == "My criticism"
    assert "Judge's concern" not in getLastUserPrompt(Llm.AllMessages[0])


def testSkepticInLaterRoundsSeesTheJudgesConcern():
    Llm = FakeLlm(Confidences=[0.9])
    Earlier = Verdict(label="MIXED", confidence=0.5, reasoning="Needs benchmarks")
    State = makeState(
        proposal="P", criticism="Old criticism", verdict=Earlier, round_number=2
    )

    Skeptic(Llm).run(State)

    Prompt = getLastUserPrompt(Llm.AllMessages[0])
    assert "Needs benchmarks" in Prompt
    assert "Old criticism" in Prompt


def testEvidenceGathererNeedsNoLlmAndReturnsEvidenceObjects():
    Update = EvidenceGatherer().run(makeState())

    assert set(Update) == {"evidence", "trace"}
    assert len(Update["evidence"]) == 2
    for Item in Update["evidence"]:
        assert isinstance(Item, Evidence)
        assert Item.is_mock is True
        assert Item.round_number == 1


def testEvidenceGathererGivesDifferentEvidenceInRoundTwo():
    RoundOne = EvidenceGatherer().gather(1)
    RoundTwo = EvidenceGatherer().gather(2)
    assert RoundOne[0].text != RoundTwo[0].text


def testEvidenceGathererSaysWhenThereIsNoMoreEvidence():
    Items = EvidenceGatherer().gather(3)
    assert len(Items) == 1
    assert "No additional evidence" in Items[0].text


def testJudgeReturnsAVerdictObject():
    Llm = FakeLlm(Confidences=[0.8])
    State = makeState(proposal="P", criticism="C")

    Update = Judge(Llm).run(State)

    assert isinstance(Update["verdict"], Verdict)
    assert Update["verdict"].confidence == 0.8
    assert Update["trace"][0].agent == "Judge"


def testJudgePromptContainsNumberedMockEvidence():
    Llm = FakeLlm(Confidences=[0.8])
    State = makeState(evidence=EvidenceGatherer().gather(1))

    Judge(Llm).run(State)

    Prompt = getLastUserPrompt(Llm.JudgeRunner.AllMessages[0])
    assert "1. [MOCK]" in Prompt
    assert "2. [MOCK]" in Prompt


def testAgentWithoutAnLlmRaisesAClearError():
    with pytest.raises(CrucibleError) as Caught:
        Proposer().run(makeState())
    assert "no LLM" in str(Caught.value)


def testLlmFailureIsWrappedWithTheAgentName():
    Llm = FakeLlm(Confidences=[0.9], ErrorToRaise=RuntimeError("network down"))

    with pytest.raises(CrucibleError) as Caught:
        Skeptic(Llm).run(makeState())

    assert "Skeptic" in str(Caught.value)
    assert "network down" in str(Caught.value)
    assert isinstance(Caught.value.__cause__, RuntimeError)


def testEmptyAnswerIsRejected():
    Llm = FakeLlm(Confidences=[0.9], TextReply="   ")

    with pytest.raises(CrucibleError) as Caught:
        Proposer(Llm).run(makeState())

    assert "empty" in str(Caught.value)


def testJudgeAsksForStrictJsonSchemaOutput():
    Llm = FakeLlm(Confidences=[0.8])

    Judge(Llm).run(makeState())

    Call = Llm.StructuredCalls[0]
    assert Call["Schema"] is Verdict
    assert Call["Options"] == {"method": "json_schema", "strict": True}
