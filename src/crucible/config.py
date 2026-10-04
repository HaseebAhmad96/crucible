"""Settings read from environment variables (and from the local .env file)."""

import os

from dotenv import load_dotenv

from crucible.errors import CrucibleError

# Reads the .env file (if one exists) into environment variables.
# Variables that are already set in the real environment are NOT overwritten.
load_dotenv()

DefaultModelName = "openai/gpt-oss-120b"
DefaultMaxRounds = 3
DefaultConfidenceThreshold = 0.75


def getApiKey() -> str:
    ApiKey = os.getenv("GROQ_API_KEY", "").strip()
    if ApiKey == "" or ApiKey == "replace-me":
        raise CrucibleError(
            "GROQ_API_KEY is not set. Copy .env.example to .env "
            "and put your real key in .env."
        )
    return ApiKey


def getModelName() -> str:
    return os.getenv("GROQ_MODEL", DefaultModelName)


def getMaxRounds() -> int:
    RawValue = os.getenv("CRUCIBLE_MAX_ROUNDS", str(DefaultMaxRounds))
    try:
        MaxRounds = int(RawValue)
    except ValueError as Error:
        raise CrucibleError(
            f"CRUCIBLE_MAX_ROUNDS must be a whole number, got '{RawValue}'."
        ) from Error
    if MaxRounds < 1:
        raise CrucibleError("CRUCIBLE_MAX_ROUNDS must be at least 1.")
    return MaxRounds


def getConfidenceThreshold() -> float:
    RawValue = os.getenv(
        "CRUCIBLE_CONFIDENCE_THRESHOLD", str(DefaultConfidenceThreshold)
    )
    try:
        Threshold = float(RawValue)
    except ValueError as Error:
        raise CrucibleError(
            f"CRUCIBLE_CONFIDENCE_THRESHOLD must be a number, got '{RawValue}'."
        ) from Error
    if not 0.0 <= Threshold <= 1.0:
        raise CrucibleError("CRUCIBLE_CONFIDENCE_THRESHOLD must be between 0 and 1.")
    return Threshold
