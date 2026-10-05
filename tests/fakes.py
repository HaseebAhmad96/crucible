"""A scripted fake LLM, so tests run instantly, free, and never touch the network."""

from langchain_core.messages import AIMessage

from crucible.schemas import Verdict


class FakeJudgeRunner:
    """Stands in for llm.with_structured_output(...). Replays scripted verdicts."""

    def __init__(self, Confidences: list[float]):
        self.Confidences = Confidences
        self.CallCount = 0
        self.AllMessages = []  # every prompt this runner received

    def invoke(self, Messages):
        self.AllMessages.append(Messages)
        # When the script runs out, keep repeating the last value.
        Position = min(self.CallCount, len(self.Confidences) - 1)
        self.CallCount += 1
        return Verdict(
            label="MIXED",
            confidence=self.Confidences[Position],
            reasoning="Fake reasoning. Still missing real evidence.",
        )


class FakeLlm:
    def __init__(
        self, Confidences: list[float], ErrorToRaise=None, TextReply="Fake text."
    ):
        self.JudgeRunner = FakeJudgeRunner(Confidences)
        self.ErrorToRaise = ErrorToRaise
        self.TextReply = TextReply
        self.TextCallCount = 0
        self.AllMessages = []  # every prompt sent for a plain-text answer
        self.StructuredCalls = []  # how with_structured_output was called

    def invoke(self, Messages):
        self.TextCallCount += 1
        self.AllMessages.append(Messages)
        if self.ErrorToRaise is not None:
            raise self.ErrorToRaise
        return AIMessage(content=self.TextReply)

    # This method name is fixed by LangChain's interface, so it stays snake_case.
    def with_structured_output(self, Schema, **Options):
        self.StructuredCalls.append({"Schema": Schema, "Options": Options})
        return self.JudgeRunner
