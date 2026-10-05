"""The Judge: weighs proposal, criticism and evidence and returns a Verdict."""

from crucible.agents.base import BaseAgent
from crucible.schemas import DebateState, Verdict


class Judge(BaseAgent):
    Name = "Judge"

    SystemPrompt = """You are the Judge in a claim-testing debate.
Read the claim, the proposal, the criticism and the evidence.
Choose a verdict label: TRUE, MOSTLY_TRUE, MIXED, MOSTLY_FALSE or FALSE.
Give a confidence from 0.0 to 1.0 for how sure you are of that label given
the evidence available. If the evidence is thin, off-topic, or leaves key
criticisms unanswered, lower your confidence.
Give a short reasoning (maximum 60 words) that also names what is still missing."""

    def run(self, State: DebateState) -> dict:
        UserPrompt = (
            f"Claim: {State.claim.text}"
            f"\n\nProposal:\n{State.proposal}"
            f"\n\nCriticism:\n{State.criticism}"
            f"\n\nEvidence:\n{self.formatEvidence(State.evidence)}"
        )
        Result = self.askStructured(Verdict, self.SystemPrompt, UserPrompt)
        Text = (
            f"Verdict: {Result.label} | Confidence: {Result.confidence:.2f}"
            f"\n{Result.reasoning}"
        )
        return {"verdict": Result, "trace": [self.makeTraceEntry(State, Text)]}
