"""Creating the LLM client and explaining LLM errors.

Agents talk to the LLM through BaseAgent.askText and BaseAgent.askStructured.
"""

from groq import AuthenticationError, RateLimitError
from langchain_groq import ChatGroq

from crucible.config import getApiKey, getModelName


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
