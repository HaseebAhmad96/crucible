"""The EvidenceGatherer. In this version it returns MOCK (hardcoded) evidence.

The mock texts below are placeholders, NOT research. This is the part that is
replaced by real tools (MCP) in v0.4.
"""

from crucible.agents.base import BaseAgent
from crucible.schemas import DebateState, Evidence

MockEvidenceByRound = {
    1: [
        "The reference Python interpreter runs bytecode, so tight numeric "
        "loops written in pure Python are usually much slower than the same loop "
        "compiled from C++.",
        "Libraries such as NumPy do their heavy numeric work in compiled "
        "code, so Python programs that use them can approach compiled speed for "
        "array operations.",
    ],
    2: [
        "When a program mostly calls optimized libraries (for example BLAS "
        "or LAPACK), the language that calls the library matters little, and C++ "
        "using the same library performs similarly.",
        "C++ is compiled ahead of time with optimization, and comparisons "
        "of equivalent hand-written algorithms generally favor C++.",
    ],
}

NoMoreEvidence = "No additional evidence is available for this round."


class EvidenceGatherer(BaseAgent):
    Name = "EvidenceGatherer"

    def gather(self, RoundNumber: int) -> list[Evidence]:
        Texts = MockEvidenceByRound.get(RoundNumber, [NoMoreEvidence])
        NewEvidence = []
        for Text in Texts:
            NewEvidence.append(
                Evidence(
                    text=Text, source="mock", round_number=RoundNumber, is_mock=True
                )
            )
        return NewEvidence

    def run(self, State: DebateState) -> dict:
        NewEvidence = self.gather(State.round_number)
        Text = self.formatEvidence(NewEvidence)
        # "evidence" accumulates (see schemas/state.py), so this list is ADDED
        # to the evidence from earlier rounds.
        return {"evidence": NewEvidence, "trace": [self.makeTraceEntry(State, Text)]}
