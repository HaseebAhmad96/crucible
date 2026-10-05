"""The DebateState model: the structured state that travels through the graph."""

import operator
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crucible.schemas.claim import Claim
from crucible.schemas.evidence import Evidence
from crucible.schemas.verdict import Verdict


class TraceEntry(BaseModel):
    """One line of the reasoning trace."""

    model_config = ConfigDict(frozen=True)

    round_number: int = Field(ge=1)
    agent: str
    text: str


class DebateState(BaseModel):
    claim: Claim
    proposal: str = ""
    criticism: str = ""
    # Annotated[..., operator.add] means: when a node returns a list for this
    # field, LangGraph APPENDS it. All other fields are simply overwritten.
    evidence: Annotated[list[Evidence], operator.add] = Field(default_factory=list)
    verdict: Verdict | None = None  # None until the Judge has spoken
    round_number: int = Field(default=1, ge=1)
    max_rounds: int = Field(default=3, ge=1)
    trace: Annotated[list[TraceEntry], operator.add] = Field(default_factory=list)

    @model_validator(mode="after")
    def checkRoundWithinLimit(self) -> "DebateState":
        if self.round_number > self.max_rounds:
            raise ValueError(
                f"round_number ({self.round_number}) is above "
                f"max_rounds ({self.max_rounds})"
            )
        return self
