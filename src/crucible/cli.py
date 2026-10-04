"""The command-line interface: read a claim, run the graph, print the trace."""

import sys
from textwrap import indent

from crucible.config import getConfidenceThreshold, getMaxRounds
from crucible.errors import CrucibleError
from crucible.graph import buildGraph
from crucible.llm import createLlm
from crucible.state import CrucibleState, TraceEntry, makeInitialState

Separator = "=" * 32


def readClaim() -> str:
    if len(sys.argv) > 1:
        Claim = " ".join(sys.argv[1:])
    else:
        print("Enter your claim:")
        Claim = input("> ")
    Claim = Claim.strip()
    if Claim == "":
        raise CrucibleError("The claim is empty. Please type a claim.")
    return Claim


def printTraceEntry(Entry: TraceEntry, LastRoundPrinted: int) -> int:
    """Print one trace entry. Returns the round number printed most recently."""
    if Entry["round_number"] != LastRoundPrinted:
        print(f"\n[Round {Entry['round_number']}]")
        LastRoundPrinted = Entry["round_number"]

    if Entry["agent"] == "Loop":
        print(f"\n🔄 {Entry['text']}")
    else:
        print(f"→ {Entry['agent']}")
        print(indent(Entry["text"], "    "))
    return LastRoundPrinted


def printFinalVerdict(State: CrucibleState, Threshold: float) -> None:
    print(f"\n{Separator}\nFINAL VERDICT\n{Separator}")
    print(f"Verdict:    {State['verdict']}")
    print(f"Confidence: {State['confidence']:.2f}")
    print(f"Rounds:     {State['round_number']} of {State['max_rounds']}")
    print(f"\nReasoning:\n{indent(State['verdict_reason'], '    ')}")
    if State["confidence"] < Threshold:
        print(
            f"\nWARNING: stopped at the round limit with confidence "
            f"below {Threshold:.2f}. Treat this verdict as weak."
        )
    print("\nNote: evidence in v0.1 is MOCK (hardcoded), so this is not research.")


def runCli() -> None:
    Claim = readClaim()
    MaxRounds = getMaxRounds()
    Threshold = getConfidenceThreshold()
    Llm = createLlm()
    Graph = buildGraph(Llm, Threshold)

    InitialState = makeInitialState(Claim, MaxRounds)
    # A second safety net: even if the router had a bug, LangGraph stops the run
    # (with GraphRecursionError) after this many steps.
    GraphConfig = {"recursion_limit": MaxRounds * 6 + 6}

    print(f"\n{Separator}\nCRUCIBLE RUN\n{Separator}")
    print(f"\nClaim:\n{Claim}")
    print("\nProcessing...")

    FinalState = InitialState
    PrintedCount = 0
    LastRoundPrinted = 0
    # stream_mode="values" gives the FULL state after every step.
    for Snapshot in Graph.stream(
        InitialState, config=GraphConfig, stream_mode="values"
    ):
        FinalState = Snapshot
        for Entry in Snapshot["trace"][PrintedCount:]:
            LastRoundPrinted = printTraceEntry(Entry, LastRoundPrinted)
        PrintedCount = len(Snapshot["trace"])

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
