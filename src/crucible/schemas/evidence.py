"""The Evidence model: one piece of evidence gathered for a claim."""

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    text: str = Field(min_length=1)
    source: str = Field(min_length=1)  # where it came from, e.g. "mock"
    round_number: int = Field(ge=1)  # the round in which it was gathered
    is_mock: bool  # required on purpose: fake evidence must always say so

    def asLine(self) -> str:
        """One readable line, with a [MOCK] tag when the evidence is fake."""
        if self.is_mock:
            return f"[MOCK] {self.text}"
        return self.text
