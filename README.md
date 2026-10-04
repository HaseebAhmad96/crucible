# Crucible

A multi-agent reasoning engine that tests a claim with four agents: Proposer, Skeptic, Evidence Gatherer and Judge. Built in versions; see the table at the bottom.

## Requirements

- Ubuntu (or WSL2 Ubuntu)
- Git
- [uv](https://docs.astral.sh/uv/)
- A Groq API key from https://console.groq.com/keys

## Setup

    git clone git@github.com:HaseebAhmad96/crucible.git
    cd crucible
    uv sync
    cp .env.example .env

Then open `.env` and replace `replace-me` with your real `GROQ_API_KEY`. The `.env` file is ignored by Git and must never be committed.

## Run

    uv run python -m crucible
    uv run python -m crucible "Your claim here"

## Test

    uv run pytest

The tests use a fake LLM, so they need no API key and make no network calls.

## How it works

    Proposer -> Skeptic -> EvidenceGatherer -> Judge
                   ^                              |
                   +---- low confidence ----------+

The Judge returns a verdict and a confidence. If confidence is below the threshold and rounds remain, the graph loops back to the Skeptic. It stops when confidence is high enough or the round limit is reached. The full reasoning trace is printed to the terminal.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| GROQ_API_KEY | (required) | Your Groq API key |
| GROQ_MODEL | openai/gpt-oss-120b | Groq model ID |
| CRUCIBLE_MAX_ROUNDS | 3 | Maximum reasoning rounds |
| CRUCIBLE_CONFIDENCE_THRESHOLD | 0.75 | Confidence needed to stop early |

## Limits of v0.1

- Evidence is hardcoded MOCK text, not research. Real evidence arrives in v0.4.
- The confidence value is the model's own estimate, not a calibrated probability.
- Free-plan Groq rate limits can interrupt long runs.

## Versions

| Version | Description |
|---|---|
| v0.0.0 | Engineering foundation (uv, Git, tests) |
| v0.1.0 | LangGraph reasoning CLI with a real LLM and mock evidence |
