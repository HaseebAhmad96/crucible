import pytest
from langchain_core.utils.function_calling import convert_to_json_schema
from pydantic import ValidationError

from crucible.schemas import Claim, DebateState, Evidence, TraceEntry, Verdict


# ---------- Claim ----------


def testClaimAcceptsNormalText():
    assert Claim(text="Python is slower than C++").text == "Python is slower than C++"


def testClaimStripsSurroundingWhitespace():
    assert Claim(text="   Water boils at 100C   ").text == "Water boils at 100C"


@pytest.mark.parametrize("BadText", ["", "     ", "Hi", "a" * 501])
def testClaimRejectsInvalidText(BadText):
    with pytest.raises(ValidationError):
        Claim(text=BadText)


def testClaimRejectsMissingText():
    with pytest.raises(ValidationError):
        Claim()


def testClaimIsReadOnly():
    UserClaim = Claim(text="A valid claim")
    with pytest.raises(ValidationError):
        UserClaim.text = "changed"


# ---------- Evidence ----------


def testEvidenceHasTheDefinedFields():
    Item = Evidence(text="Some fact", source="mock", round_number=1, is_mock=True)
    assert Item.text == "Some fact"
    assert Item.source == "mock"
    assert Item.round_number == 1
    assert Item.is_mock is True


def testEvidenceAsLineMarksMockEvidence():
    Fake = Evidence(text="Some fact", source="mock", round_number=1, is_mock=True)
    Real = Evidence(text="Some fact", source="web", round_number=1, is_mock=False)
    assert Fake.asLine() == "[MOCK] Some fact"
    assert Real.asLine() == "Some fact"


def testEvidenceRequiresIsMock():
    with pytest.raises(ValidationError):
        Evidence(text="Some fact", source="mock", round_number=1)


@pytest.mark.parametrize(
    "Changes",
    [{"text": ""}, {"source": ""}, {"round_number": 0}],
)
def testEvidenceRejectsBadFields(Changes):
    Fields = {"text": "Fact", "source": "mock", "round_number": 1, "is_mock": True}
    Fields.update(Changes)
    with pytest.raises(ValidationError):
        Evidence(**Fields)


# ---------- Verdict ----------


def testVerdictHasTheDefinedFields():
    Result = Verdict(label="TRUE", confidence=0.8, reasoning="ok")
    assert Result.label == "TRUE"
    assert Result.confidence == 0.8
    assert Result.reasoning == "ok"


@pytest.mark.parametrize("BadConfidence", [-0.1, 1.5])
def testVerdictRejectsConfidenceOutsideZeroToOne(BadConfidence):
    with pytest.raises(ValidationError):
        Verdict(label="TRUE", confidence=BadConfidence, reasoning="ok")


def testVerdictRejectsUnknownLabel():
    with pytest.raises(ValidationError):
        Verdict(label="MAYBE", confidence=0.5, reasoning="ok")


def testVerdictJsonSchemaIsStrictModeCompatible():
    # This is the schema that is sent to Groq. Strict mode needs every property
    # listed as required and additionalProperties set to false.
    Schema = convert_to_json_schema(Verdict, strict=True)
    assert Schema["additionalProperties"] is False
    assert set(Schema["required"]) == {"label", "confidence", "reasoning"}


# ---------- DebateState ----------


def makeClaim() -> Claim:
    return Claim(text="A valid claim")


def testDebateStateDefaults():
    State = DebateState(claim=makeClaim())
    assert State.round_number == 1
    assert State.max_rounds == 3
    assert State.verdict is None
    assert State.evidence == []
    assert State.trace == []
    assert State.proposal == ""


def testDebateStateListsAreNotSharedBetweenInstances():
    First = DebateState(claim=makeClaim())
    Second = DebateState(claim=makeClaim())
    First.trace.append(TraceEntry(round_number=1, agent="Proposer", text="x"))
    assert Second.trace == []


def testDebateStateRequiresAClaim():
    with pytest.raises(ValidationError):
        DebateState()


def testDebateStateRejectsAStringInsteadOfAClaim():
    with pytest.raises(ValidationError):
        DebateState(claim="just a string")


@pytest.mark.parametrize("Field", ["round_number", "max_rounds"])
def testDebateStateRejectsZeroRounds(Field):
    with pytest.raises(ValidationError):
        DebateState(claim=makeClaim(), **{Field: 0})


def testDebateStateRejectsRoundAboveMaxRounds():
    with pytest.raises(ValidationError):
        DebateState(claim=makeClaim(), round_number=4, max_rounds=3)


def testDebateStateAcceptsAVerdict():
    Result = Verdict(label="FALSE", confidence=0.9, reasoning="ok")
    State = DebateState(claim=makeClaim(), verdict=Result)
    assert State.verdict.label == "FALSE"
