import crucible


def testPackageExposesMain():
    assert callable(crucible.main)
