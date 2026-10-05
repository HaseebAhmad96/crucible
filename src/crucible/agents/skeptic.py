"""The Skeptic: looks for weaknesses, assumptions and missing information."""

from crucible.agents.base import BaseAgent
from crucible.schemas import DebateState


class Skeptic(BaseAgent):
    Name = "Skeptic"

    SystemPrompt = """You are the Skeptic in a claim-testing debate.
Find weaknesses in the proposal: hidden assumptions, overgeneralizations,
missing information, and what evidence would settle the question. Be specific.
Maximum 120 words. Plain text, no headings."""

    def run(self, State: DebateState) -> dict:
        UserPrompt = f"Claim: {State.claim.text}\n\nProposal:\n{State.proposal}"
        # In later rounds, the Skeptic also sees what the Judge said was missing.
        if State.round_number > 1 and State.verdict is not None:
            UserPrompt += (
                f"\n\nYour previous criticism:\n{State.criticism}"
                f"\n\nThe Judge's concern after the last round:\n{State.verdict.reasoning}"
                "\n\nFocus on what is still missing or unsupported."
            )
        Reply = self.askText(self.SystemPrompt, UserPrompt)
        return {"criticism": Reply, "trace": [self.makeTraceEntry(State, Reply)]}
