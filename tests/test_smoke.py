from crucible import main
def testMain(capsys):
  main()
  output = capsys.readouterr().out
  assert "crucible" in output.lower()
