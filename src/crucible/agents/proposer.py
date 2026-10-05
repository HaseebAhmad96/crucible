"""The Proposer: states the strongest reasonable position on the claim."""

from crucible.agents.base import BaseAgent
from crucible.schemas import DebateState


class Proposer(BaseAgent):
    Name = "Proposer"

    SystemPrompt = """You are the Proposer in a claim-testing debate.
State the strongest reasonable position on the claim, including important
conditions or limits of scope. Be concrete.
Maximum 120 words. Plain text, no headings."""

    def run(self, State: DebateState) -> dict:
        Reply = self.askText(self.SystemPrompt, f"Claim: {State.claim.text}")
        return {"proposal": Reply, "trace": [self.makeTraceEntry(State, Reply)]}
