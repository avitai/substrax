"""The framework facts the NNX rules and checks rest on, measured on the installed flax and jax."""

from __future__ import annotations

import dataclasses

import pytest

from substrax.testing.nnx_premises import (
    check_premises,
    measure_premises,
    PremiseChangedError,
    PREMISES,
)
from substrax.testing.nnx_rules import RULES


def test_every_premise_holds_on_the_installed_frameworks() -> None:
    check_premises()


def test_each_premise_names_what_rests_on_it() -> None:
    """A premise that changes must say which rule or check to revisit."""
    known = set(RULES) | {"tree-mode", "conformance"}
    for premise in PREMISES:
        assert premise.relied_on_by, premise.name
        assert set(premise.relied_on_by) <= known, premise.name
        assert len(premise.why.split()) >= 8, premise.name


def test_measurements_report_every_premise_by_name() -> None:
    assert set(measure_premises()) == {premise.name for premise in PREMISES}
    assert len(PREMISES) == len({premise.name for premise in PREMISES})


def test_a_changed_premise_is_named_with_what_rests_on_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: a premise whose measurement flips fails, naming itself and its rules."""
    first = PREMISES[0]
    flipped = dataclasses.replace(first, measure=lambda: not first.measure())
    monkeypatch.setattr("substrax.testing.nnx_premises.PREMISES", (flipped, *PREMISES[1:]))

    with pytest.raises(PremiseChangedError) as raised:
        check_premises()

    message = str(raised.value)
    assert first.name in message
    assert all(rule in message for rule in first.relied_on_by)
