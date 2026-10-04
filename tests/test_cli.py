import pytest
from fakes import FakeLlm

from crucible import cli


def testCliPrintsTraceAndFinalVerdict(monkeypatch, capsys):
    monkeypatch.setattr(cli, "createLlm", lambda: FakeLlm(Confidences=[0.50, 0.90]))
    monkeypatch.setattr("sys.argv", ["crucible", "A test claim"])

    cli.main()

    Output = capsys.readouterr().out
    assert "CRUCIBLE RUN" in Output
    assert "[Round 1]" in Output
    assert "[Round 2]" in Output
    assert "→ Proposer" in Output
    assert "→ Judge" in Output
    assert "FINAL VERDICT" in Output
    assert "MOCK" in Output


def testCliExitsWithErrorWhenLlmFails(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "createLlm",
        lambda: FakeLlm(Confidences=[0.9], ErrorToRaise=RuntimeError("boom")),
    )
    monkeypatch.setattr("sys.argv", ["crucible", "A test claim"])

    with pytest.raises(SystemExit) as Caught:
        cli.main()

    assert Caught.value.code == 1
    assert "ERROR" in capsys.readouterr().err
