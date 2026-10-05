"""BaseAgent: the behavior that every agent shares."""

from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from crucible.errors import CrucibleError
from crucible.llm import explainError
from crucible.schemas import DebateState, Evidence, TraceEntry


class BaseAgent(ABC):
    # Subclasses overwrite this. It is used in the trace and in error messages.
    Name = "BaseAgent"

    def __init__(self, Llm=None):
        # Llm is optional: the EvidenceGatherer does not need one (yet).
        self.Llm = Llm

    @abstractmethod
    def run(self, State: DebateState) -> dict:
        """Read the state and return ONLY the fields this agent changes."""

    def makeTraceEntry(self, State: DebateState, Text: str) -> TraceEntry:
        return TraceEntry(round_number=State.round_number, agent=self.Name, text=Text)

    @staticmethod
    def formatEvidence(EvidenceList: list[Evidence]) -> str:
        Lines = []
        for Number, Item in enumerate(EvidenceList, start=1):
            Lines.append(f"{Number}. {Item.asLine()}")
        return "\n".join(Lines)

    def requireLlm(self):
        if self.Llm is None:
            raise CrucibleError(f"{self.Name} has no LLM configured.")
        return self.Llm

    def askText(self, SystemPrompt: str, UserPrompt: str) -> str:
        Llm = self.requireLlm()
        Messages = [SystemMessage(content=SystemPrompt), HumanMessage(content=UserPrompt)]
        try:
            Response = Llm.invoke(Messages)
        except Exception as Error:
            # We never swallow the error: we add context and raise it again.
            raise CrucibleError(
                f"{self.Name} could not get an answer from the LLM. "
                f"{explainError(Error)}"
            ) from Error

        Text = Response.content
        if not isinstance(Text, str) or Text.strip() == "":
            raise CrucibleError(
                f"{self.Name} got an empty answer from the LLM. "
                "If this repeats, try raising max_tokens in createLlm()."
            )
        return Text.strip()

    def askStructured(
        self, Schema: type[BaseModel], SystemPrompt: str, UserPrompt: str
    ) -> BaseModel:
        Llm = self.requireLlm()
        Messages = [SystemMessage(content=SystemPrompt), HumanMessage(content=UserPrompt)]
        StructuredLlm = Llm.with_structured_output(
            Schema, method="json_schema", strict=True
        )
        try:
            Result = StructuredLlm.invoke(Messages)
        except Exception as Error:
            raise CrucibleError(
                f"{self.Name} could not produce valid structured output. "
                f"{explainError(Error)}"
            ) from Error

        if not isinstance(Result, Schema):
            raise CrucibleError(
                f"{self.Name} got something that is not a {Schema.__name__}."
            )
        return Result
