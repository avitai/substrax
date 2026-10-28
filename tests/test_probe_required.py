"""A planted failure: the required check must go red on it. This branch is never merged."""


def test_planted_failure() -> None:
    assert 1 == 2  # noqa: PLR0133
