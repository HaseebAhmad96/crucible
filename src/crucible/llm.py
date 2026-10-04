"""Everything that talks to the LLM lives here, so errors are handled in one place."""

from typing import Literal

from groq import AuthenticationError, RateLimitError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator

from crucible.config import getApiKey, getModelName
from crucible.errors import CrucibleError


class JudgeOutput(BaseModel):
    """The structured answer we require from the Judge."""

    verdict: Literal["TRUE", "MOSTLY_TRUE", "MIXED", "MOSTLY_FALSE", "FALSE"] = Field(
        description="Overall judgement of the claim."
    )
    confidence: float = Field(
        description="How sure you are of the verdict, from 0.0 to 1.0."
    )
    reasoning: str = Field(
        description="Short reasoning, also naming what evidence is still missing."
    )

    # The 0-to-1 rule is checked here in Python instead of in the JSON schema,
    # because strict structured output may not accept min/max constraints.
    @field_validator("confidence")
    @classmethod
    def checkConfidenceRange(cls, Value: float) -> float:
        if not 0.0 <= Value <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {Value}")
        return Value


def createLlm() -> ChatGroq:
    return ChatGroq(
        model=getModelName(),
        api_key=getApiKey(),
        temperature=0.2,
        max_tokens=1500,
        max_retries=2,
        timeout=60,
        reasoning_effort="low",
    )


def explainError(Error: Exception) -> str:
    if isinstance(Error, RateLimitError):
        return (
            "Groq rate limit reached (free-plan limits are small). "
            "Wait a minute and run again."
        )
    if isinstance(Error, AuthenticationError):
        return "Groq rejected the API key. Check GROQ_API_KEY in .env."
    return f"{type(Error).__name__}: {Error}"


def askText(Llm, SystemPrompt: str, UserPrompt: str, AgentName: str) -> str:
    Messages = [SystemMessage(content=SystemPrompt), HumanMessage(content=UserPrompt)]
    try:
        Response = Llm.invoke(Messages)
    except Exception as Error:
        # We never swallow the error: we add context and raise it again.
        raise CrucibleError(
            f"{AgentName} could not get an answer from the LLM. {explainError(Error)}"
        ) from Error

    Text = Response.content
    if not isinstance(Text, str) or Text.strip() == "":
        raise CrucibleError(
            f"{AgentName} got an empty answer from the LLM. "
            "If this repeats, try raising max_tokens in createLlm()."
        )
    return Text.strip()


def askJudge(Llm, SystemPrompt: str, UserPrompt: str) -> JudgeOutput:
    Messages = [SystemMessage(content=SystemPrompt), HumanMessage(content=UserPrompt)]
    JudgeLlm = Llm.with_structured_output(
        JudgeOutput, method="json_schema", strict=True
    )
    try:
        Result = JudgeLlm.invoke(Messages)
    except Exception as Error:
        raise CrucibleError(
            f"Judge could not produce a valid structured verdict. {explainError(Error)}"
        ) from Error

    if not isinstance(Result, JudgeOutput):
        raise CrucibleError("Judge returned something that is not a JudgeOutput.")
    return Result
