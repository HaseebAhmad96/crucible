import pytest

from crucible.config import getApiKey, getConfidenceThreshold, getMaxRounds
from crucible.errors import CrucibleError


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
