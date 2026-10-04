"""The graph nodes. Each node reads the state and returns ONLY the fields it changes."""

from crucible.evidence import getMockEvidence
from crucible.llm import askJudge, askText
from crucible.state import CrucibleState, makeTraceEntry

ProposerPrompt = """You are the Proposer in a claim-testing debate.
State the strongest reasonable position on the claim, including important
conditions or limits of scope. Be concrete.
Maximum 120 words. Plain text, no headings."""

SkepticPrompt = """You are the Skeptic in a claim-testing debate.
Find weaknesses in the proposal: hidden assumptions, overgeneralizations,
missing information, and what evidence would settle the question. Be specific.
Maximum 120 words. Plain text, no headings."""

JudgePrompt = """You are the Judge in a claim-testing debate.
Read the claim, the proposal, the criticism and the evidence.
Choose a verdict: TRUE, MOSTLY_TRUE, MIXED, MOSTLY_FALSE or FALSE.
Give a confidence from 0.0 to 1.0 for how sure you are of that verdict given
the evidence available. If the evidence is thin, off-topic, or leaves key
criticisms unanswered, lower your confidence.
Give a short reasoning (maximum 60 words) that also names what is still missing."""


def formatEvidence(EvidenceList: list[str]) -> str:
    Lines = []
    for Number, Item in enumerate(EvidenceList, start=1):
        Lines.append(f"{Number}. {Item}")
    return "\n".join(Lines)


def proposerNode(State: CrucibleState, Llm) -> dict:
    Reply = askText(Llm, ProposerPrompt, f"Claim: {State['claim']}", "Proposer")
    Entry = makeTraceEntry(State["round_number"], "Proposer", Reply)
    return {"proposal": Reply, "trace": [Entry]}


def skepticNode(State: CrucibleState, Llm) -> dict:
    UserPrompt = f"Claim: {State['claim']}\n\nProposal:\n{State['proposal']}"
    if State["round_number"] > 1:
        UserPrompt += (
            f"\n\nYour previous criticism:\n{State['criticism']}"
            f"\n\nThe Judge's concern after the last round:\n{State['verdict_reason']}"
            "\n\nFocus on what is still missing or unsupported."
        )
    Reply = askText(Llm, SkepticPrompt, UserPrompt, "Skeptic")
    Entry = makeTraceEntry(State["round_number"], "Skeptic", Reply)
    return {"criticism": Reply, "trace": [Entry]}


def evidenceGathererNode(State: CrucibleState) -> dict:
    NewEvidence = getMockEvidence(State["round_number"])
    Entry = makeTraceEntry(
        State["round_number"], "EvidenceGatherer", formatEvidence(NewEvidence)
    )
    # "evidence" accumulates (see state.py), so this list is ADDED to earlier rounds.
    return {"evidence": NewEvidence, "trace": [Entry]}


def judgeNode(State: CrucibleState, Llm) -> dict:
    UserPrompt = (
        f"Claim: {State['claim']}"
        f"\n\nProposal:\n{State['proposal']}"
        f"\n\nCriticism:\n{State['criticism']}"
        f"\n\nEvidence:\n{formatEvidence(State['evidence'])}"
    )
    Result = askJudge(Llm, JudgePrompt, UserPrompt)
    Text = (
        f"Verdict: {Result.verdict} | Confidence: {Result.confidence:.2f}"
        f"\n{Result.reasoning}"
    )
    Entry = makeTraceEntry(State["round_number"], "Judge", Text)
    return {
        "verdict": Result.verdict,
        "confidence": Result.confidence,
        "verdict_reason": Result.reasoning,
        "trace": [Entry],
    }


def nextRoundNode(State: CrucibleState) -> dict:
    NewRound = State["round_number"] + 1
    Text = (
        f"Confidence {State['confidence']:.2f} is too low. "
        f"Looping back to the Skeptic (round {NewRound} of {State['max_rounds']})."
    )
    # This entry belongs to the round that just ended, so it keeps the OLD number.
    Entry = makeTraceEntry(State["round_number"], "Loop", Text)
    return {"round_number": NewRound, "trace": [Entry]}
