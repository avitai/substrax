"""Source rules for Flax NNX library code, each with the shapes it must catch and must leave."""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

from substrax.testing import run_python
from substrax.testing.nnx_rules import (
    exemptions,
    findings,
    MalformedExemptionError,
    RULES,
)


def _sites(rule: str, source: str) -> list[int]:
    return [site.line for site in RULES[rule](ast.parse(source))]


def _write(directory: Path, name: str, source: str) -> Path:
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


# Shapes the seed rule must catch: DiffBio's and datarax's copies of the rule, merged.
SEED_LITERALS = [
    "a = nnx.Rngs(0)",
    "b = rngs or nnx.Rngs(0)",
    "c = nnx.Rngs(params=0, sample=1)",
    "d = jax.random.key(0)",
    "e = jax.random.PRNGKey(42)",
    "f = random.key(3)",
    "g = np.random.default_rng(7)",
    "h = np.random.RandomState(1)",
    "i = np.random.Philox(2)",
    "np.random.seed(3)",
    "seed = 0",
    "seed: int = 42",
    "f(seed=7)",
    "self.seed = 5",
    "def f(x, seed=0): ...",
    "def f(x, *, seed: int = 1): ...",
    "k = jax.random.key(\n    0\n)",
]
# Shapes it must leave: a seed the caller chose, a seed passed on, prose.
NOT_SEED_LITERALS = [
    "a = nnx.Rngs(seed)",
    "b = jax.random.key(config.seed)",
    "c = rngs_from_seed(0, streams)",
    '"""Example: Pipeline(..., rngs=nnx.Rngs(0))."""',
    "# seed = 3",
    "random_seed = 0",
    "seed_value = 3",
    "is_seeded = seed == 0",
    "f(seed=config.seed)",
    "def f(x, seed=None): ...",
    "n = sorted(items, key=len)",
]


@pytest.mark.parametrize("source", SEED_LITERALS)
def test_a_seed_written_into_library_code_is_found(source: str) -> None:
    assert len(_sites("seed-literal", source)) == 1


@pytest.mark.parametrize("source", NOT_SEED_LITERALS)
def test_a_seed_the_caller_chose_is_not_a_literal(source: str) -> None:
    assert _sites("seed-literal", source) == []


@pytest.mark.parametrize(
    ("rule", "caught", "spared"),
    [
        (
            "rngs-probe",
            [
                "'sample' in rngs",
                "'dropout' not in self.rngs",
                "hasattr(rngs, 'params')",
                "getattr(self._rngs, 'sample', None)",
                "rngs.get('sample')",
            ],
            [
                "rngs.sample()",
                "'x' in options",
                "hasattr(module, 'rngs')",
                "'x' in strings",
                "hasattr(substrings, 'a')",
            ],
        ),
        (
            "rngs-fallback",
            [
                "a = rngs or nnx.Rngs(0)",
                "b = rngs if rngs is not None else nnx.Rngs(params=key)",
                "if rngs is None:\n    rngs = nnx.Rngs(0)",
            ],
            [
                "a = rngs or default_rngs",
                "if rngs is None:\n    raise ValueError('rngs is required')",
            ],
        ),
        (
            "trainable-metadata",
            ["t = nnx.Param(table, trainable=False)", "t = nnx.Param(table, trainable=True)"],
            ["t = nnx.Param(table)", "t = nnx.Variable(table, trainable=False)"],
        ),
        (
            "salted-hash",
            ["token = hash(word) % vocab_size"],
            [
                "class A:\n    def __hash__(self):\n        return hash((self.a, self.b))",
                "digest = hashlib.sha256(data)",
            ],
        ),
        (
            "mode-override",
            [
                "def __call__(self, x, deterministic: bool = False): ...",
                "def g(x, *, training=True): ...",
                "def h(x, use_running_average: bool = True): ...",
                "def k(x, train=False): ...",
                "def f(x, deterministic: bool | None = False): ...",
                "def set_view(self, deterministic: bool = True): ...",
            ],
            [
                "def f(x, deterministic: bool | None = None): ...",
                "def f(x, deterministic): ...",
                "def g(x, *, verbose: bool = False): ...",
                "def __init__(self, deterministic: bool = False):\n"
                "    self.deterministic = deterministic",
            ],
        ),
        (
            "unread-mode-flag",
            [
                "def __call__(self, x, deterministic=None):\n    return self.linear(x)",
                "def __call__(self, x, *, training: bool = False):\n    del training\n    return x",
                "def __init__(self, deterministic=False):\n    self.rate = 0.5",
            ],
            [
                "def __call__(self, x, deterministic=None):\n"
                "    return self.dropout(x, deterministic=deterministic)",
                'def __call__(self, x, deterministic=None):\n    """An interface."""\n    ...',
                "def __call__(self, x, deterministic=None):\n    raise NotImplementedError",
                "def f(x, training):\n    def g():\n        return training\n    return g()",
            ],
        ),
        (
            "unreachable-mode",
            [
                "class A(nnx.Module):\n    def __init__(self):\n        self.training = True",
                "def __init__(self, deterministic=False):\n    self._deterministic = deterministic",
                "def __init__(self, deterministic=False):\n"
                "    if not deterministic:\n        self.dropout = nnx.Dropout(0.5)",
                "def __init__(self, training=True):\n    self.mode = training",
                "def reset(self):\n    self.is_training = False",
                "model.training = False",
            ],
            [
                "def __init__(self, deterministic=False):\n    self.deterministic = deterministic",
                "def __init__(self, rate, deterministic=False):\n"
                "    self.dropout = nnx.Dropout(rate, deterministic=deterministic)",
                "def __init__(self, training=True):\n"
                "    self.norm = nnx.BatchNorm(4, use_running_average=not training)",
                "def __init__(self, deterministic=False):\n"
                "    super().__init__(deterministic=deterministic)",
                "def __init__(self, deterministic=False):\n"
                "    self.deterministic: bool = deterministic",
                "def set_view(self, deterministic=None):\n"
                "    if deterministic is not None:\n        self.deterministic = deterministic",
                "model.deterministic = True",
            ],
        ),
        (
            "noop-super-init",
            ["class A(nnx.Module):\n    def __init__(self):\n        super().__init__()"],
            [
                "class B(nnx.Linear):\n    def __init__(self):\n        super().__init__(2, 3)",
                "class C(Base):\n    def __init__(self):\n        super().__init__()",
            ],
        ),
    ],
)
def test_each_rule_catches_its_shapes_and_spares_the_rest(
    rule: str, caught: list[str], spared: list[str]
) -> None:
    for source in caught:
        assert len(_sites(rule, source)) == 1, source
    for source in spared:
        assert _sites(rule, source) == [], source


def test_every_rule_is_exercised_above() -> None:
    """A rule added without its controls fails here."""
    assert set(RULES) == {
        "seed-literal",
        "rngs-probe",
        "rngs-fallback",
        "trainable-metadata",
        "salted-hash",
        "mode-override",
        "unread-mode-flag",
        "unreachable-mode",
        "noop-super-init",
    }


def test_an_exemption_names_its_rule_and_a_reason() -> None:
    text = (
        "key = jax.random.key(0)  # substrax-allow: seed-literal: shape-only dummy input\n"
        "cache = hash(value)  # substrax-allow: salted-hash: in-process cache key\n"
    )

    assert exemptions(text) == {
        1: ("seed-literal", "shape-only dummy input"),
        2: ("salted-hash", "in-process cache key"),
    }


@pytest.mark.parametrize(
    "comment",
    [
        "# substrax-allow: seed-literal",
        "# substrax-allow: seed-literal:   ",
        "# substrax-allow: no-such-rule: a reason",
        "# substrax-allow seed-literal: a reason",
    ],
)
def test_an_exemption_without_a_known_rule_or_a_reason_is_refused(comment: str) -> None:
    with pytest.raises(MalformedExemptionError):
        exemptions(f"key = jax.random.key(0)  {comment}\n")


def test_findings_skip_exempted_sites_and_report_the_rest(tmp_path: Path) -> None:
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    _write(
        package,
        "module.py",
        "key = jax.random.key(0)  # substrax-allow: seed-literal: shape-only dummy input\n"
        "other = jax.random.key(1)\n"
        "token = hash(word)\n",
    )

    found = findings(tmp_path, ("src/pkg",))

    assert [(f.path, f.line, f.rule) for f in found] == [
        ("src/pkg/module.py", 2, "seed-literal"),
        ("src/pkg/module.py", 3, "salted-hash"),
    ]
    assert found[0].source == "jax.random.key(1)"


def test_an_exemption_a_formatter_moved_to_the_closing_line_still_applies(tmp_path: Path) -> None:
    """ruff format wraps a long call and leaves its trailing comment on the closing line."""
    _write(
        tmp_path,
        "module.py",
        "table = nnx.Param(\n"
        "    jnp.ones(3), trainable=False\n"
        ")  # substrax-allow: trainable-metadata: the premise measures it\n",
    )

    assert findings(tmp_path, (".",)) == []


def test_an_exemption_only_covers_the_rule_it_names(tmp_path: Path) -> None:
    _write(tmp_path, "module.py", "a = hash(nnx.Rngs(0))  # substrax-allow: salted-hash: cache\n")

    assert [f.rule for f in findings(tmp_path, (".",))] == ["seed-literal"]


def test_findings_can_be_limited_to_chosen_rules(tmp_path: Path) -> None:
    _write(tmp_path, "module.py", "a = nnx.Rngs(0)\nb = hash(word)\n")

    assert [f.rule for f in findings(tmp_path, (".",), rules=("salted-hash",))] == ["salted-hash"]


def test_an_unknown_rule_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="no-such-rule"):
        findings(tmp_path, (".",), rules=("no-such-rule",))


def test_a_missing_directory_fails_instead_of_shrinking_the_corpus(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        findings(tmp_path, ("src/missing",))


def test_a_literal_planted_in_a_copy_of_this_package_is_found(tmp_path: Path) -> None:
    """Positive control on a real corpus: this package is clean, and one planted seed is found."""
    package = Path(__file__).resolve().parents[2] / "src" / "substrax"
    copy = tmp_path / "src" / "substrax"
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
    planted = copy / "rng" / "keys.py"
    planted.write_text(planted.read_text() + "\n_ORDER = np.random.default_rng(7)\n")
    lines = len(planted.read_text().splitlines())

    assert findings(Path(__file__).resolve().parents[2], ("src/substrax",)) == []
    assert [(f.path, f.line) for f in findings(tmp_path, ("src/substrax",))] == [
        ("src/substrax/rng/keys.py", lines)
    ]


def test_importing_the_rules_imports_no_jax() -> None:
    result = run_python(
        "import sys, substrax.testing.nnx_rules\n"
        "from substrax.testing import report\n"
        "report('jax' in sys.modules)",
        timeout=60,
    )

    assert result.check().report_as(bool) is False
