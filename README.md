# Crucible

A multi-agent reasoning engine that tests claims through a Proposer, Skeptic, Evidence Gatherer and Judge. Work in progress, built in versions.

## Requirements

- Ubuntu (or WSL2 Ubuntu)
- Git
- [uv](https://docs.astral.sh/uv/)

## Setup

    git clone git@github.com:HaseebAhmad96/crucible.git
    cd crucible
    uv sync

## Run

    uv run python -m crucible

## Test

    uv run pytest

## Environment variables

Secrets live in a local `.env` file, which is never committed. `.env.example` is the template. It is not needed until v0.1, which adds LLM calls.

## Versions

| Version | Description |
|---|---|
| v0.0.0 | Engineering foundation (uv, Git, tests) |
| v0.1.0 | Planned: LangGraph reasoning CLI |
## Versions

