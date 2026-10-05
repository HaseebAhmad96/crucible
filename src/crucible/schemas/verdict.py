"""The Verdict model: the Judge's structured answer.

This same class is (1) the JSON schema sent to the LLM and (2) the object stored
in the graph state, so what the LLM returns is already validated data.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

VerdictLabel = Literal["TRUE", "MOSTLY_TRUE", "MIXED", "MOSTLY_FALSE", "FALSE"]


class Verdict(BaseModel):
    """The structured answer we require from the Judge."""

    model_config = ConfigDict(frozen=True)

    label: VerdictLabel = Field(description="Overall judgement of the claim.")
    confidence: float = Field(
        description="How sure you are of the label, from 0.0 to 1.0."
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
