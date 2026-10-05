"""The Claim model: the statement the user wants tested."""

from pydantic import BaseModel, ConfigDict, Field


class Claim(BaseModel):
    # frozen=True makes the object read-only after creation.
    # str_strip_whitespace=True trims spaces BEFORE the length rules are checked.
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    text: str = Field(min_length=5, max_length=500)
