import pytest
from pydantic import ValidationError

from crucible.config import getApiKey, getConfidenceThreshold, getMaxRounds
from crucible.errors import CrucibleError
from crucible.llm import JudgeOutput


def testJudgeOutputAcceptsValidData():
    Result = JudgeOutput(verdict="TRUE", confidence=0.8, reasoning="ok")
    assert Result.confidence == 0.8


@pytest.mark.parametrize("BadConfidence", [-0.1, 1.5])
def testJudgeOutputRejectsConfidenceOutsideZeroToOne(BadConfidence):
    with pytest.raises(ValidationError):
        JudgeOutput(verdict="TRUE", confidence=BadConfidence, reasoning="ok")


def testJudgeOutputRejectsUnknownVerdict():
    with pytest.raises(ValidationError):
        JudgeOutput(verdict="MAYBE", confidence=0.5, reasoning="ok")


def testMissingApiKeyRaisesClearError(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(CrucibleError) as Caught:
        getApiKey()
    assert "GROQ_API_KEY" in str(Caught.value)


def testPlaceholderApiKeyIsRejected(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "replace-me")
    with pytest.raises(CrucibleError):
        getApiKey()


def testBadMaxRoundsRaisesClearError(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_MAX_ROUNDS", "three")
    with pytest.raises(CrucibleError):
        getMaxRounds()


def testBadThresholdRaisesClearError(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_CONFIDENCE_THRESHOLD", "1.5")
    with pytest.raises(CrucibleError):
        getConfidenceThreshold()
