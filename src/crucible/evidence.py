"""MOCK evidence. Hardcoded placeholder text, NOT real research.

This file is replaced by real tools (MCP) in v0.4. Every line is tagged [MOCK]
so it can never be mistaken for a verified source.
"""

MockEvidenceByRound = {
    1: [
        "[MOCK] The reference Python interpreter runs bytecode, so tight numeric "
        "loops written in pure Python are usually much slower than the same loop "
        "compiled from C++.",
        "[MOCK] Libraries such as NumPy do their heavy numeric work in compiled "
        "code, so Python programs that use them can approach compiled speed for "
        "array operations.",
    ],
    2: [
        "[MOCK] When a program mostly calls optimized libraries (for example BLAS "
        "or LAPACK), the language that calls the library matters little, and C++ "
        "using the same library performs similarly.",
        "[MOCK] C++ is compiled ahead of time with optimization, and comparisons "
        "of equivalent hand-written algorithms generally favor C++.",
    ],
}

NoMoreEvidence = ["[MOCK] No additional evidence is available for this round."]


def getMockEvidence(RoundNumber: int) -> list[str]:
    return MockEvidenceByRound.get(RoundNumber, NoMoreEvidence)
