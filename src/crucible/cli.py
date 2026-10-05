"""The command-line interface: read a claim, run the graph, print the trace."""

import sys
from textwrap import indent

from pydantic import ValidationError

from crucible.config import getConfidenceThreshold, getMaxRounds
from crucible.errors import CrucibleError
from crucible.graph import buildGraph
from crucible.llm import createLlm
from crucible.schemas import Claim, DebateState, TraceEntry

Separator = "=" * 32


def describeValidationError(Error: ValidationError) -> str:
    Messages = []
    for Detail in Error.errors():
        Messages.append(Detail["msg"])
    return "; ".join(Messages)


def readClaim() -> Claim:
    if len(sys.argv) > 1:
        RawText = " ".join(sys.argv[1:])
    else:
        print("Enter your claim:")
        RawText = input("> ")

    # The Claim model is the gatekeeper: nothing invalid gets past this line.
    try:
        return Claim(text=RawText)
    except ValidationError as Error:
        raise CrucibleError(
            f"Invalid claim: {describeValidationError(Error)}"
        ) from Error


def validateFinalState(Snapshot: dict) -> DebateState:
    """LangGraph returns a plain dict, so we turn it back into a checked DebateState."""
    try:
        return DebateState.model_validate(Snapshot)
    except ValidationError as Error:
        raise CrucibleError(
            f"The final state is invalid: {describeValidationError(Error)}"
        ) from Error


def printTraceEntry(Entry: TraceEntry, LastRoundPrinted: int) -> int:
    """Print one trace entry. Returns the round number printed most recently."""
    if Entry.round_number != LastRoundPrinted:
        print(f"\n[Round {Entry.round_number}]")
        LastRoundPrinted = Entry.round_number

    if Entry.agent == "Loop":
        print(f"\n🔄 {Entry.text}")
    else:
        print(f"→ {Entry.agent}")
        print(indent(Entry.text, "    "))
    return LastRoundPrinted


def printFinalVerdict(State: DebateState, Threshold: float) -> None:
    FinalVerdict = State.verdict
    if FinalVerdict is None:
        raise CrucibleError("The run finished without a verdict.")

    print(f"\n{Separator}\nFINAL VERDICT\n{Separator}")
    print(f"Verdict:    {FinalVerdict.label}")
    print(f"Confidence: {FinalVerdict.confidence:.2f}")
    print(f"Rounds:     {State.round_number} of {State.max_rounds}")
    print(f"\nReasoning:\n{indent(FinalVerdict.reasoning, '    ')}")
    if FinalVerdict.confidence < Threshold:
        print(
            f"\nWARNING: stopped at the round limit with confidence "
            f"below {Threshold:.2f}. Treat this verdict as weak."
        )
    print("\nNote: evidence is MOCK (hardcoded) in this version, so this is not research.")


def runCli() -> None:
    UserClaim = readClaim()
    MaxRounds = getMaxRounds()
    Threshold = getConfidenceThreshold()
    Llm = createLlm()
    Graph = buildGraph(Llm, Threshold)

    InitialState = DebateState(claim=UserClaim, max_rounds=MaxRounds)
    # A second safety net: even if the router had a bug, LangGraph stops the run
    # (with GraphRecursionError) after this many steps.
    GraphConfig = {"recursion_limit": MaxRounds * 6 + 6}

    print(f"\n{Separator}\nCRUCIBLE RUN\n{Separator}")
    print(f"\nClaim:\n{UserClaim.text}")
    print("\nProcessing...")

    LastSnapshot = {}
    PrintedCount = 0
    LastRoundPrinted = 0
    # stream_mode="values" gives the FULL state (as a dict) after every step.
    for Snapshot in Graph.stream(
        InitialState, config=GraphConfig, stream_mode="values"
    ):
        LastSnapshot = Snapshot
        TraceList = Snapshot["trace"]
        for Entry in TraceList[PrintedCount:]:
            LastRoundPrinted = printTraceEntry(Entry, LastRoundPrinted)
        PrintedCount = len(TraceList)

    FinalState = validateFinalState(LastSnapshot)
    printFinalVerdict(FinalState, Threshold)


def main() -> None:
    try:
        runCli()
    except CrucibleError as Error:
        print(f"\nERROR: {Error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        sys.exit(130)
