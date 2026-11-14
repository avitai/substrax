# tests/examples/test_normalising.py
from typing import TypedDict

from substrax.testing.example_suite import ExampleRunner


class Normalised(TypedDict):
    count: int
    mean_error: float
    std_error: float


def test_the_whole_batch_is_normalised(example_run: ExampleRunner) -> None:
    normalised = example_run("statistics/normalising.py").returned_as(Normalised)
    assert normalised["count"] == 128
    assert normalised["mean_error"] < 1e-6
