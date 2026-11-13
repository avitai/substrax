"""``substrax.notebooks.conformance``: an example follows the shared example guide."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from _example_texts import CONFIG, GOOD, GPU_REQUIRED, write_example

from substrax.devices import info, require_gpu
from substrax.notebooks.conformance import check_example, example_metadata, Finding, RULES
from substrax.notebooks.pages import generate_page


STATIC = frozenset(RULES) - {"page"}


def _rules_broken(tmp_path: Path, text: str) -> list[str]:
    repo, script = write_example(tmp_path, text)
    return sorted(
        {
            finding.rule
            for finding in check_example(script, repo_root=repo, config=CONFIG, rules=STATIC)
        }
    )


def test_every_rule_has_an_id_a_title_and_a_check() -> None:
    assert RULES
    for rule_id, rule in RULES.items():
        assert rule.id == rule_id
        assert rule.title
        assert callable(rule.check)


def test_a_conforming_example_breaks_no_static_rule(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)

    assert check_example(script, repo_root=repo, config=CONFIG, rules=STATIC) == []


def test_a_conforming_example_with_its_page_breaks_no_rule(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)
    generate_page(script, repo_root=repo, config=CONFIG)

    assert check_example(script, repo_root=repo, config=CONFIG) == []


def test_a_missing_page_breaks_the_page_rule(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)

    rules = {
        finding.rule
        for finding in check_example(script, repo_root=repo, config=CONFIG, rules={"page"})
    }

    assert rules == {"page"}


def test_the_page_rule_holds_the_page_to_the_settings_it_is_given(tmp_path: Path) -> None:
    """The caller's settings decide the page's limit, not a second read of pyproject.toml."""
    repo, script = write_example(tmp_path)
    generate_page(script, repo_root=repo, config=CONFIG)
    tight = dataclasses.replace(CONFIG, page_size_warn_bytes=1, page_size_bytes=1)

    findings = check_example(script, repo_root=repo, config=tight, rules={"page"})

    assert [finding.detail.split(":")[0] for finding in findings] == ["too-large"]


@pytest.mark.parametrize(
    ("change", "rule"),
    [
        (("x", "x"), None),
        (("# Sorting a Batch of Records", "Sorting a Batch of Records"), "title"),
        (("| **Tier** | Quick Reference |", "| **Tier** | Beginner |"), "metadata"),
        (("| **Devices** | CPU-compatible |", "| **Devices** | GPU |"), "metadata"),
        (("| **Prerequisites** | Basic Python |\n", ""), "metadata"),
        (
            (
                "| **Prerequisites** | Basic Python |",
                "| **Prerequisites** | Basic Python |\n| **Runtime** | ~5 min |",
            ),
            "metadata",
        ),
        (("## Overview", "## Summary"), "skeleton"),
        (("## Learning Goals", "## What You'll Learn"), "skeleton"),
        (("## Results\n", "## Results & Evaluation\n"), "skeleton"),
        (("## Next Steps", "## Further Reading"), "skeleton"),
        (("1. Sort a batch", "- Sort a batch"), "goals"),
        (("### Step 2: Sort them", "### Step 3: Sort them"), "steps"),
        (("### Step 1: Make the records", "### Part 1: Make the records"), "steps"),
        (('if __name__ == "__main__":\n    main()', "main()"), "main"),
        (
            (
                'if __name__ == "__main__":\n    main()',
                'if __name__ == "__main__":\n    summary = main()\n    print(summary)',
            ),
            "main",
        ),
        (("def main() -> None:", "def run() -> None:"), "main"),
        (
            ("import jax\n", 'import os\n\nos.environ["JAX_PLATFORMS"] = "cpu"\nimport jax\n'),
            "no-backend-forcing",
        ),
        (
            ("import jax\n", 'import jax\n\njax.config.update("jax_platform_name", "cpu")\n'),
            "no-backend-forcing",
        ),
        (
            ("print(sort_records(make_records()))", 'open("docs/assets/out.txt", "w").write("x")'),
            "artifacts",
        ),
        (
            (
                "import jax\n",
                'import jax\nimport matplotlib.pyplot as plt\n\nplt.savefig("out.png")\n',
            ),
            "artifacts",
        ),
        *(
            (
                ("Sorts records by key.", f"Sorts records by key. {instruction}"),
                "no-backend-forcing",
            )
            for instruction in (
                "Run `export JAX_PLATFORMS=cpu` first.",
                "Run `JAX_PLATFORMS=cpu uv run python examples/sorting.py`.",
                "Run `unset JAX_PLATFORMS` first.",
                "Set `CUDA_VISIBLE_DEVICES=0` for one GPU.",
                '\n\n```python\njax.config.update("jax_platform_name", "cpu")\n```\n',
            )
        ),
        (
            ("Sorts records by key.", "Sorts records by key. The lane sets JAX_PLATFORMS."),
            None,
        ),
        (("import jax\n", "import jax\nfrom math import *\n"), "no-star-imports"),
        (("jax.random.key(0)", "jax.random.PRNGKey(0)"), "typed-keys"),
    ],
)
def test_each_violation_breaks_exactly_its_rule(
    tmp_path: Path, change: tuple[str, str], rule: str | None
) -> None:
    old, new = change
    assert old in GOOD

    broken = _rules_broken(tmp_path, GOOD.replace(old, new))

    assert broken == ([] if rule is None else [rule])


def test_a_missing_section_is_one_finding_not_one_per_rule(tmp_path: Path) -> None:
    text = GOOD.replace("## Implementation\n\n", "")

    assert _rules_broken(tmp_path, text) == ["skeleton"]


def test_main_and_its_guard_may_share_the_last_cell(tmp_path: Path) -> None:
    text = GOOD.replace(
        "    print(sort_records(make_records()))\n\n\n# %%\nif __name__",
        "    print(sort_records(make_records()))\n\n\nif __name__",
    )
    assert text != GOOD

    assert _rules_broken(tmp_path, text) == []


def test_optional_sections_in_their_places_are_allowed(tmp_path: Path) -> None:
    # Core Concepts comes after Setup's code cell, which holds the imports, and before
    # Implementation.
    text = GOOD.replace(
        '"""\n## Implementation',
        '"""\n## Core Concepts\n\nWhat sorting means.\n\n## Implementation',
    ).replace("## Next Steps", "## Troubleshooting\n\n### Unsorted output\n\n## Next Steps")

    assert _rules_broken(tmp_path, text) == []


def test_optional_sections_out_of_place_break_the_skeleton(tmp_path: Path) -> None:
    text = GOOD.replace("## Next Steps", "## Next Steps\n\n## Troubleshooting")

    assert _rules_broken(tmp_path, text) == ["skeleton"]


def test_a_docstring_closing_a_markdown_cell_early_breaks_parsing(tmp_path: Path) -> None:
    text = GOOD.replace("Sorts records by key.", 'Sorts records by key with """sorted""" calls.')

    assert "parses" in _rules_broken(tmp_path, text)


def test_a_notebook_out_of_step_with_its_script_breaks_the_pair_rule(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)
    script.write_text(
        GOOD.replace("sorted(records)", "sorted(records, reverse=True)"), encoding="utf-8"
    )

    rules = {
        finding.rule
        for finding in check_example(script, repo_root=repo, config=CONFIG, rules=STATIC)
    }

    assert rules == {"pair"}


def test_a_finding_names_its_rule_and_script(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path, GOOD.replace("jax.random.key(0)", "jax.random.PRNGKey(0)")
    )

    finding = check_example(script, repo_root=repo, config=CONFIG, rules={"typed-keys"})[0]

    assert isinstance(finding, Finding)
    assert finding.script == script
    assert "typed-keys" in str(finding)
    assert str(script) in str(finding)


def test_an_unknown_rule_is_refused(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)

    with pytest.raises(KeyError, match="no-such-rule"):
        check_example(script, repo_root=repo, config=CONFIG, rules={"no-such-rule"})


def test_the_metadata_table_is_read_field_by_field(tmp_path: Path) -> None:
    _, script = write_example(tmp_path)

    assert example_metadata(script) == {
        "Tier": "Quick Reference",
        "Devices": "CPU-compatible",
        "Prerequisites": "Basic Python",
    }


def _findings(tmp_path: Path, text: str) -> list[Finding]:
    repo, script = write_example(tmp_path, text)
    return check_example(script, repo_root=repo, config=CONFIG, rules=STATIC)


def test_a_numbered_list_inside_fenced_code_is_not_the_learning_goals(tmp_path: Path) -> None:
    text = GOOD.replace(
        "1. Sort a batch of records by key\n2. Check the order of the result\n",
        "The goals are below.\n\n```text\n1. not a goal\n```\n",
    )

    assert _rules_broken(tmp_path, text) == ["goals"]


def test_a_file_opening_with_a_byte_order_mark_is_one_parses_finding(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)
    script.write_bytes(b"\xef\xbb\xbf" + script.read_bytes())

    findings = check_example(script, repo_root=repo, config=CONFIG, rules=STATIC)

    assert [finding.rule for finding in findings] == ["parses"]
    assert "byte order mark" in findings[0].detail


def test_an_async_main_is_named_by_the_main_rule(tmp_path: Path) -> None:
    findings = _findings(tmp_path, GOOD.replace("def main() -> None:", "async def main() -> None:"))

    assert [(finding.rule, "async" in finding.detail) for finding in findings] == [("main", True)]


def test_a_markdown_cell_written_as_comments_breaks_parses(tmp_path: Path) -> None:
    text = GOOD.replace('# %% [markdown]\n"""\n## Setup\n"""', "# %% [markdown]\n# ## Setup")
    assert text != GOOD

    findings = _findings(tmp_path, text)

    assert [finding.rule for finding in findings] == ["parses"]
    assert "triple-quoted" in findings[0].detail


def test_a_file_that_is_not_utf8_is_a_finding_through_every_source_rule(tmp_path: Path) -> None:
    # The hook, the command line and the shared tests run every source rule, `pair` included.
    repo, script = write_example(tmp_path)
    script.write_bytes(script.read_bytes().replace(b"Sorts records", b"Sorts r\xe9cords"))

    findings = check_example(script, repo_root=repo, config=CONFIG, rules=STATIC)

    utf8 = {finding.rule for finding in findings if "UTF-8" in finding.detail}
    assert utf8 == {"parses", "pair"}


_SETUP_CODE = '# %%\nimport jax\n\n# %% [markdown]\n"""\n## Implementation'


def _with_setup(code: str) -> str:
    """The GPU-required example with ``code`` as its Setup cell."""
    assert _SETUP_CODE in GPU_REQUIRED
    return GPU_REQUIRED.replace(
        _SETUP_CODE, f'# %%\n{code}\n\n# %% [markdown]\n"""\n## Implementation'
    )


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        pytest.param(GPU_REQUIRED, "gpu-check", id="no-check"),
        pytest.param(
            _with_setup("import jax\n\nfrom substrax.devices import require_gpu\n\nrequire_gpu()"),
            None,
            id="imported",
        ),
        pytest.param(
            _with_setup("import jax\n\nimport substrax.devices\n\nsubstrax.devices.require_gpu()"),
            None,
            id="through-the-module",
        ),
        pytest.param(
            GPU_REQUIRED.replace(
                "    return sorted(records)",
                "    from substrax.devices import require_gpu\n\n    require_gpu()\n"
                "    return sorted(records)",
            ),
            "gpu-check",
            id="outside-setup",
        ),
        pytest.param(
            _with_setup("import jax\n\n\ndef require_gpu() -> None:\n    pass\n\n\nrequire_gpu()"),
            "gpu-check",
            id="homemade",
        ),
        pytest.param(
            GOOD.replace(
                "import jax\n", "import jax\n\nfrom substrax.devices import require_gpu\n"
            ),
            None,
            id="cpu-compatible-needs-none",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices.info import require_gpu\n\nrequire_gpu()"
            ),
            None,
            id="from-its-own-module",
        ),
        pytest.param(
            _with_setup("import jax\n\nimport substrax.devices.info as info\n\ninfo.require_gpu()"),
            None,
            id="through-its-own-module",
        ),
        pytest.param(
            _with_setup("import jax\n\nfrom substrax.devices import info\n\ninfo.require_gpu()"),
            None,
            id="its-module-from-the-package",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n\ndevices = require_gpu()"
            ),
            None,
            id="assigned",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n\n\n"
                "def check() -> None:\n    require_gpu()"
            ),
            "gpu-check",
            id="in-a-function-never-called",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n\n\n"
                "def check() -> None:\n    require_gpu()\n\n\ncheck()"
            ),
            "gpu-check",
            id="through-a-function",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n\n"
                'if jax.default_backend() != "cpu":\n    require_gpu()'
            ),
            "gpu-check",
            id="under-a-condition",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n\n"
                "try:\n    require_gpu()\nexcept RuntimeError:\n    pass"
            ),
            "gpu-check",
            id="its-refusal-caught",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n\n"
                "batch = jax.numpy.ones(3)\nrequire_gpu()"
            ),
            "gpu-check",
            id="after-other-work",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n"
                "from os import getcwd as require_gpu\n\nrequire_gpu()"
            ),
            "gpu-check",
            id="rebound-by-a-later-import",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n\n"
                '"""No work runs before the check."""\nrequire_gpu()'
            ),
            None,
            id="after-a-docstring",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n\nrequire_gpu()\n"
                "from os import getcwd as require_gpu"
            ),
            None,
            id="rebound-after-the-call",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n\n"
                "try:\n    import numpy\nexcept ImportError:\n    numpy = None\n\nrequire_gpu()"
            ),
            "gpu-check",
            id="after-a-try-import",
        ),
        pytest.param(
            _with_setup("import jax\n\nsubstrax.devices.require_gpu()"),
            "gpu-check",
            id="its-package-never-imported",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nimport substrax.devices\nimport numpy as substrax\n\n"
                "substrax.devices.require_gpu()"
            ),
            "gpu-check",
            id="its-package-rebound",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n\nrequire_gpu(object())"
            ),
            "gpu-check",
            id="handed-devices-of-its-own",
        ),
        pytest.param(
            _with_setup(
                "import jax\n\nfrom substrax.devices import require_gpu\n\nrequire_gpu(info=None)"
            ),
            "gpu-check",
            id="given-a-keyword",
        ),
    ],
)
def test_a_gpu_required_example_checks_for_its_gpu_in_setup(
    tmp_path: Path, text: str, rule: str | None
) -> None:
    assert _rules_broken(tmp_path, text) == ([] if rule is None else [rule])


def test_gpu_check_reads_require_gpu_where_it_lives() -> None:
    """The rule names substrax.devices and the module defining require_gpu; both reach it."""
    assert require_gpu is info.require_gpu
    assert info.require_gpu.__module__ == "substrax.devices.info"


@pytest.mark.parametrize(
    ("change", "rule"),
    [
        pytest.param(
            ("import jax\n", "import jax\nfrom jax.random import PRNGKey\n"),
            "typed-keys",
            id="imported",
        ),
        pytest.param(
            ("import jax\n", "import jax\nfrom jax.random import PRNGKey as make_key\n"),
            "typed-keys",
            id="imported-under-another-name",
        ),
        pytest.param(
            ("import jax\n", "import jax\nfrom jax import random\n"),
            None,
            id="the-module-alone",
        ),
        pytest.param(
            ("jax.random.key(0)", "jax.random.key(0)  # not jax.random.PRNGKey"),
            None,
            id="named-in-a-comment",
        ),
    ],
)
def test_typed_keys_covers_prngkey_however_it_is_reached(
    tmp_path: Path, change: tuple[str, str], rule: str | None
) -> None:
    old, new = change
    assert old in GOOD

    assert _rules_broken(tmp_path, GOOD.replace(old, new)) == ([] if rule is None else [rule])


@pytest.mark.parametrize(
    ("statement", "rule"),
    [
        pytest.param('os.environ["X_SMOKE"] = "1"', "no-env-writes", id="assign"),
        pytest.param('os.environ["X_SMOKE"] += "1"', "no-env-writes", id="augmented"),
        pytest.param('os.environ.setdefault("X_SMOKE", "1")', "no-env-writes", id="setdefault"),
        pytest.param('os.environ.update(X_SMOKE="1")', "no-env-writes", id="update"),
        pytest.param('os.environ |= {"X_SMOKE": "1"}', "no-env-writes", id="or-assign"),
        pytest.param('os.environ.pop("X_SMOKE", None)', "no-env-writes", id="pop"),
        pytest.param('del os.environ["X_SMOKE"]', "no-env-writes", id="delete"),
        pytest.param('os.putenv("X_SMOKE", "1")', "no-env-writes", id="putenv"),
        pytest.param('os.unsetenv("X_SMOKE")', "no-env-writes", id="unsetenv"),
        pytest.param('environ["X_SMOKE"] = "1"', "no-env-writes", id="imported-environ"),
        pytest.param('print(os.environ.get("X_SMOKE"))', None, id="a-read"),
        pytest.param('print(os.environ["HOME"])', None, id="a-subscript-read"),
        pytest.param(
            'os.environ.setdefault("JAX_PLATFORMS", "cpu")', "no-backend-forcing", id="backend"
        ),
    ],
)
def test_an_example_reads_the_environment_and_never_writes_it(
    tmp_path: Path, statement: str, rule: str | None
) -> None:
    """A write of a backend setting is no-backend-forcing's finding alone: one finding a write."""
    setup = "import os\nfrom os import environ\n\nimport jax\n"
    text = GOOD.replace("import jax\n", setup).replace(
        "    print(sort_records(make_records()))",
        f"    {statement}\n    print(sort_records(make_records()))",
    )

    assert _rules_broken(tmp_path, text) == ([] if rule is None else [rule])


_PRINT = "    print(sort_records(make_records()))"
_RESULTS = "## Results\n\n### Sorted order\n"


def _commented(comment: str) -> str:
    return GOOD.replace(_PRINT, f"    {comment}\n{_PRINT}")


def _results_line(line: str) -> str:
    return GOOD.replace(_RESULTS, f"{_RESULTS}\n{line}\n")


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        # A line that opens by stating output: the census's forms and the concept's others.
        pytest.param(
            _commented("# Expected output: [0, 1, 2, 3, 4]"), "no-expected-output", id="output"
        ),
        pytest.param(
            _commented("# Expected: the records sorted"), "no-expected-output", id="expected"
        ),
        pytest.param(
            _commented("# Expected output (approximate): [0, 1, 2]"),
            "no-expected-output",
            id="qualified",
        ),
        pytest.param(
            _commented("# Expected (with checkpoint): a sorted list"),
            "no-expected-output",
            id="note",
        ),
        pytest.param(
            _commented("# Expected output in quick mode:"), "no-expected-output", id="in-a-mode"
        ),
        pytest.param(
            _commented("# ## Expected Output"), "no-expected-output", id="commented-heading"
        ),
        pytest.param(
            GOOD.replace(_PRINT, f"{_PRINT}  # Expected output: [0, 1, 2, 3, 4]"),
            "no-expected-output",
            id="trailing",
        ),
        pytest.param(_results_line("### Expected Output"), "no-expected-output", id="heading"),
        pytest.param(
            _results_line("### \N{CLAPPER BOARD} Expected Output"), "no-expected-output", id="emoji"
        ),
        pytest.param(
            _results_line("Expected: the records come back sorted."),
            "no-expected-output",
            id="a-prose-line",
        ),
        pytest.param(_commented("# Sample output:"), "no-expected-output", id="sample"),
        pytest.param(_results_line("Example output:"), "no-expected-output", id="example-output"),
        # Text that only mentions output, or names something else. A bare `Output` names what code
        # returns, a shape or a figure (OWN-1110-SHAPE-COMMENTS-ALLOWED); every claim measured in
        # the stack's examples opens with Expected (OWN-1112-EXPECTED-OUTPUT-OPENS-WITH-EXPECTED).
        pytest.param(
            _commented("# Output: logits of shape (batch, classes)"), None, id="a-shape-comment"
        ),
        pytest.param(_commented("# Output (batch, 10) after the head"), None, id="a-bare-shape"),
        pytest.param(
            _results_line("Output (left) and target (right) are compared in the figure."),
            None,
            id="a-figure-reference",
        ),
        pytest.param(_commented("# Output: [0, 1, 2, 3, 4]"), None, id="output-colon"),
        pytest.param(_commented("# Output (approximate): a sorted list"), None, id="output-note"),
        pytest.param(_commented("# Output directory for the run's files"), None, id="a-noun"),
        pytest.param(
            _commented("# Compare each step with the expected output"), None, id="a-mention"
        ),
        pytest.param(
            GOOD.replace(_PRINT, '    print("# Expected output: sorted")\n' + _PRINT),
            None,
            id="inside-a-string",
        ),
        pytest.param(
            _results_line("The run prints the expected output of each step."), None, id="in-prose"
        ),
    ],
)
def test_an_example_shows_its_run_never_an_expected_output(
    tmp_path: Path, text: str, rule: str | None
) -> None:
    assert _rules_broken(tmp_path, text) == ([] if rule is None else [rule])


@pytest.mark.parametrize(
    ("relative", "rule"),
    [
        pytest.param("sorting.py", None, id="a-name"),
        pytest.param("data/sort_records_v2.py", None, id="a-topic-and-digits-inside"),
        pytest.param("01_sorting.py", "names", id="number-prefix"),
        pytest.param("2d/sorting.py", "names", id="a-topic-with-a-number-prefix"),
        pytest.param("Sorting.py", "names", id="capitals"),
        pytest.param("Data/sorting.py", "names", id="a-topic-in-capitals"),
        pytest.param("sorting-records.py", "names", id="hyphens"),
        pytest.param("sorting__records.py", "names", id="a-double-underscore"),
        pytest.param("sorting_.py", "names", id="a-trailing-underscore"),
    ],
)
def test_every_part_of_an_examples_path_is_snake_case_with_no_number_prefix(
    tmp_path: Path, relative: str, rule: str | None
) -> None:
    repo, script = write_example(tmp_path, GOOD, relative=relative)

    broken = sorted(
        {
            finding.rule
            for finding in check_example(script, repo_root=repo, config=CONFIG, rules=STATIC)
        }
    )

    assert broken == ([] if rule is None else [rule])


@pytest.mark.parametrize(
    ("title", "rule"),
    [
        pytest.param("Sorting a Batch of Records", None, id="the-subject"),
        pytest.param("A Guide to Sorting Records", None, id="guide-alone-is-no-tier"),
        pytest.param("Sorting Records: a Quick Reference", "title", id="quick-reference"),
        pytest.param("Tutorial: Sorting Records", "title", id="tutorial"),
        pytest.param("An advanced guide to sorting", "title", id="advanced-guide-any-case"),
    ],
)
def test_the_title_names_the_subject_not_the_tier(
    tmp_path: Path, title: str, rule: str | None
) -> None:
    text = GOOD.replace("# Sorting a Batch of Records", f"# {title}")

    assert _rules_broken(tmp_path, text) == ([] if rule is None else [rule])


_HEADER = "| Metadata | Value |\n| --- | --- |\n"


@pytest.mark.parametrize(
    ("change", "rule"),
    [
        pytest.param((_HEADER, "| Metadata | Value |\n| :--- | :---: |\n"), None, id="aligned"),
        pytest.param((_HEADER, ""), "metadata", id="no-header"),
        pytest.param(
            (_HEADER, "| Field | Value |\n| --- | --- |\n"), "metadata", id="other-labels"
        ),
        pytest.param((_HEADER, "| Metadata | Value |\n"), "metadata", id="no-delimiter"),
        pytest.param(
            ("# Sorting a Batch of Records\n\n", "# Sorting a Batch of Records\n\nAn intro.\n\n"),
            "metadata",
            id="prose-before-the-table",
        ),
    ],
)
def test_the_metadata_table_with_its_header_directly_follows_the_title(
    tmp_path: Path, change: tuple[str, str], rule: str | None
) -> None:
    old, new = change
    assert old in GOOD

    assert _rules_broken(tmp_path, GOOD.replace(old, new)) == ([] if rule is None else [rule])


_IMPLEMENTATION_CODE = "# %%\ndef make_records() -> list[int]:"
_NEXT_STEPS = "## Next Steps\n\n- [Another example](another.md)\n"
_TROUBLESHOOTING = "## Troubleshooting\n\n"


@pytest.mark.parametrize(
    ("change", "rule"),
    [
        pytest.param(
            (
                _IMPLEMENTATION_CODE,
                "# %%\nimport math\n\n\n" + _IMPLEMENTATION_CODE.removeprefix("# %%\n"),
            ),
            "setup",
            id="an-import-outside-setup",
        ),
        pytest.param(
            ("# %%\nimport jax\n", "# %%\nimport jax\n\n# %%\nfrom pathlib import Path\n"),
            "setup",
            id="setup-in-two-cells",
        ),
        pytest.param(
            (
                "    return sorted(records)",
                "    from operator import itemgetter\n\n"
                "    del itemgetter\n    return sorted(records)",
            ),
            None,
            id="an-import-inside-a-function",
        ),
        pytest.param(
            ("### Sorted order\n", "The records come back sorted.\n"),
            "results",
            id="results-untitled",
        ),
        pytest.param(
            (_NEXT_STEPS, f"{_TROUBLESHOOTING}A slow run means a large batch.\n\n{_NEXT_STEPS}"),
            "troubleshooting",
            id="troubleshooting-untitled",
        ),
        pytest.param(
            (
                _NEXT_STEPS,
                f"{_TROUBLESHOOTING}### The run is slow\n\nA large batch.\n\n{_NEXT_STEPS}",
            ),
            None,
            id="troubleshooting-entries",
        ),
        pytest.param(
            (_NEXT_STEPS, f"{_NEXT_STEPS}\n### Experiments to Try\n\n- Sort in reverse.\n"),
            None,
            id="next-steps-experiments",
        ),
        pytest.param(
            (_NEXT_STEPS, f"{_NEXT_STEPS}\n### Further Reading\n\n- A paper.\n"),
            "next-steps",
            id="next-steps-another-subsection",
        ),
    ],
)
def test_each_section_holds_what_the_guide_gives_it(
    tmp_path: Path, change: tuple[str, str], rule: str | None
) -> None:
    old, new = change
    assert old in GOOD

    assert _rules_broken(tmp_path, GOOD.replace(old, new)) == ([] if rule is None else [rule])


@pytest.mark.parametrize(
    ("imports", "statement", "rule"),
    [
        pytest.param("import sys", "print(sys.argv[0])", None, id="the-scripts-own-name"),
        pytest.param("import sys", "print(sys.argv[1])", "no-arguments", id="an-argument"),
        pytest.param("import sys", "print(len(sys.argv))", "no-arguments", id="their-count"),
        pytest.param("import sys", "print(sys.argv[1:])", "no-arguments", id="a-slice"),
        pytest.param("from sys import argv", "print(argv[1])", "no-arguments", id="imported-argv"),
        pytest.param("import argparse", "print(argparse.__name__)", "no-arguments", id="argparse"),
        pytest.param(
            "from argparse import ArgumentParser",
            "print(ArgumentParser)",
            "no-arguments",
            id="from-argparse",
        ),
        pytest.param("import getopt", "print(getopt.__name__)", "no-arguments", id="getopt"),
        pytest.param("import optparse", "print(optparse.__name__)", "no-arguments", id="optparse"),
        pytest.param("import click", "print(click)", "no-arguments", id="click"),
        pytest.param("import typer", "print(typer)", "no-arguments", id="typer"),
    ],
)
def test_an_example_reads_no_command_line_arguments(
    tmp_path: Path, imports: str, statement: str, rule: str | None
) -> None:
    text = GOOD.replace("import jax\n", f"{imports}\n\nimport jax\n").replace(
        "    print(sort_records(make_records()))",
        f"    {statement}\n    print(sort_records(make_records()))",
    )

    assert _rules_broken(tmp_path, text) == ([] if rule is None else [rule])


@pytest.mark.parametrize(
    ("statement", "rule"),
    [
        pytest.param("print(jax.devices())", None, id="listing-the-devices"),
        pytest.param("print(len(jax.local_devices()))", None, id="counting-local-devices"),
        pytest.param("print(jax.device_put(1.0))", None, id="placing-on-jaxs-choice"),
        pytest.param('print(jax.devices("cpu"))', "no-backend-forcing", id="devices-by-platform"),
        pytest.param(
            'print(jax.devices(backend="gpu"))', "no-backend-forcing", id="devices-by-keyword"
        ),
        pytest.param(
            'print(jax.local_devices(backend="cpu"))',
            "no-backend-forcing",
            id="local-devices-by-platform",
        ),
        pytest.param(
            'print(jax.jit(sorted, backend="cpu"))', "no-backend-forcing", id="a-backend-keyword"
        ),
        pytest.param(
            "with jax.default_device(jax.devices()[0]):\n        print(1)",
            "no-backend-forcing",
            id="a-default-device",
        ),
        pytest.param(
            'jax.config.update("jax_default_device", jax.devices()[0])',
            "no-backend-forcing",
            id="the-default-device-setting",
        ),
    ],
)
def test_an_example_never_chooses_the_device_jax_runs_on(
    tmp_path: Path, statement: str, rule: str | None
) -> None:
    text = GOOD.replace(
        "    print(sort_records(make_records()))",
        f"    {statement}\n    print(sort_records(make_records()))",
    )

    assert _rules_broken(tmp_path, text) == ([] if rule is None else [rule])


def test_a_function_s_own_import_of_jax_is_not_followed(tmp_path: Path) -> None:
    """A name means what the module's imports bind it to, not how it is spelled."""
    assert GOOD.count("# %%\nimport jax\n") == 1
    text = GOOD.replace("# %%\nimport jax\n", "# %%\nimport numpy\n").replace(
        "    print(sort_records(make_records()))",
        '    import jax\n\n    print(jax.devices("cpu"))\n    print(sort_records(make_records()))',
    )

    assert "no-backend-forcing" not in _rules_broken(tmp_path, text)


@pytest.mark.parametrize(
    ("imports", "statement", "rule"),
    [
        pytest.param(
            "from jax import devices", 'print(devices("cpu"))', "no-backend-forcing", id="by-name"
        ),
        pytest.param(
            "from jax import devices as listed",
            'print(listed("cpu"))',
            "no-backend-forcing",
            id="renamed",
        ),
        pytest.param(
            "import jax as j", 'print(j.devices("cpu"))', "no-backend-forcing", id="module-alias"
        ),
        pytest.param(
            "from jax import local_devices",
            'print(local_devices(None, "cpu"))',
            "no-backend-forcing",
            id="local-devices-backend-positional",
        ),
        pytest.param(
            "import jax as j",
            "with j.default_device(j.devices()[0]):\n        print(1)",
            "no-backend-forcing",
            id="default-device-through-an-alias",
        ),
        pytest.param("from jax import devices", "print(devices())", None, id="listing-by-name"),
        pytest.param(
            "from jax import local_devices", "print(local_devices(0))", None, id="a-process-index"
        ),
        pytest.param(
            "from os import getcwd as devices",
            'print(devices("cpu"))',
            None,
            id="another-modules-name",
        ),
        pytest.param(
            "from jax import devices\nfrom os import getcwd as devices",
            'print(devices("cpu"))',
            None,
            id="taken-back-by-a-later-import",
        ),
        pytest.param(
            "from jax import devices",
            "def helper() -> None:\n        from os import getcwd as devices\n\n"
            '        print(devices())\n\n    print(devices("cpu"))',
            "no-backend-forcing",
            id="a-function-s-own-import-leaves-the-module-s",
        ),
        pytest.param(
            "import numpy as jax", 'print(jax.devices("cpu"))', None, id="jax-rebound-to-another"
        ),
        pytest.param("", "print(jax.devices(None))", None, id="the-default-backend-s-devices"),
        pytest.param(
            "", "print(jax.local_devices(0, None))", None, id="the-default-backend-s-local-devices"
        ),
        pytest.param(
            "", "print(jax.devices(backend=None))", None, id="the-default-backend-by-keyword"
        ),
        pytest.param(
            "try:\n    import jax as guarded\nexcept ImportError:\n    guarded = None",
            'print(guarded.devices("cpu"))',
            "no-backend-forcing",
            id="imported-under-a-module-level-try",
        ),
        pytest.param(
            "import sys\n\nif sys.platform:\n    import jax as chosen",
            'print(chosen.devices("cpu"))',
            "no-backend-forcing",
            id="imported-under-a-module-level-if",
        ),
        pytest.param(
            "import contextlib\n\nwith contextlib.suppress(ImportError):\n"
            "    from jax import devices as listed",
            'print(listed("cpu"))',
            "no-backend-forcing",
            id="imported-under-a-module-level-with",
        ),
        pytest.param(
            "class Holder:\n    import jax as inner",
            'print(inner.devices("cpu"))',
            None,
            id="a-class-body-has-a-scope-of-its-own",
        ),
    ],
)
def test_a_device_choice_is_found_through_any_import_of_jax(
    tmp_path: Path, imports: str, statement: str, rule: str | None
) -> None:
    """``jax.local_devices(process_index, backend)`` takes the backend second."""
    text = GOOD.replace("import jax\n", f"import jax\n{imports}\n", 1).replace(
        "    print(sort_records(make_records()))",
        f"    {statement}\n    print(sort_records(make_records()))",
    )

    assert _rules_broken(tmp_path, text) == ([] if rule is None else [rule])


@pytest.mark.parametrize(
    "call",
    [
        pytest.param("main(steps=5)", id="a-keyword"),
        pytest.param("main(5)", id="a-positional"),
        pytest.param("main(*settings)", id="unpacked"),
        pytest.param("main(**settings)", id="unpacked-keywords"),
    ],
)
def test_the_guard_calls_main_with_no_arguments(tmp_path: Path, call: str) -> None:
    """The run reads main()'s value and its exit from the guard's bare main() call only."""
    text = GOOD.replace(
        'if __name__ == "__main__":\n    main()', f'if __name__ == "__main__":\n    {call}'
    )

    assert _rules_broken(tmp_path, text) == ["main"]


_SETUP_CELL = '# %% [markdown]\n"""\n## Setup\n"""'


@pytest.mark.parametrize(
    ("cell", "named"),
    [
        pytest.param('# %% [markdown]\nr"""\n## Setup\n"""', None, id="a-raw-triple-quoted-string"),
        pytest.param("# %% [md]\n# ## Setup", "comments", id="the-md-alias-as-comments"),
        pytest.param("# %% [markdown]\n# ## Setup", "comments", id="comments"),
        pytest.param('# %% [markdown]\n"## Setup"', "a one-line string", id="a-one-line-string"),
    ],
)
def test_a_markdown_cell_is_a_triple_quoted_string_and_named_for_what_it_is(
    tmp_path: Path, cell: str, named: str | None
) -> None:
    assert _SETUP_CELL in GOOD
    repo, script = write_example(tmp_path, GOOD.replace(_SETUP_CELL, cell))

    details = [
        finding.detail
        for finding in check_example(script, repo_root=repo, config=CONFIG, rules={"parses"})
        if "markdown cell" in finding.detail
    ]

    assert len(details) == (0 if named is None else 1)
    assert all(f"written as {named}" in detail for detail in details)


_SETUP_IMPORTS = "# %%\nimport jax"


@pytest.mark.parametrize(
    ("marker", "refused"),
    [
        pytest.param("# %%", False, id="a-plain-marker"),
        pytest.param("# %% Import the library", False, id="a-title"),
        pytest.param('# %% tags=["skip-execution"]', True, id="a-tag-the-run-obeys"),
        pytest.param('# %% active="py"', True, id="a-cell-only-the-script-runs"),
        pytest.param('# %% active="ipynb"', True, id="a-cell-only-the-notebook-runs"),
        pytest.param("# %% [raw]", True, id="a-raw-cell"),
        pytest.param('# %% some_key="value"', True, id="any-other-option"),
    ],
)
def test_a_cell_marker_carries_no_option_the_script_run_ignores(
    tmp_path: Path, marker: str, refused: bool
) -> None:
    """A cell option changes what the notebook runs and never what ``python <script>`` runs."""
    assert GOOD.count(_SETUP_IMPORTS) == 1
    text = GOOD.replace(_SETUP_IMPORTS, f"{marker}\nimport jax")
    line = text.splitlines().index(marker) + 1

    findings = _findings(tmp_path, text)

    assert [finding.rule for finding in findings] == (["parses"] if refused else [])
    assert all(f"line {line}: " in finding.detail for finding in findings)


@pytest.mark.parametrize("marker", ["# %% [markdown] The setup", "# %% [md]"])
def test_a_titled_or_aliased_markdown_cell_is_a_plain_cell(tmp_path: Path, marker: str) -> None:
    assert _SETUP_CELL in GOOD
    text = GOOD.replace(_SETUP_CELL, f'{marker}\n"""\n## Setup\n"""')

    assert _findings(tmp_path, text) == []


@pytest.mark.parametrize(
    "block",
    [
        pytest.param("```text\n## Setup\n```", id="a-backtick-fence"),
        pytest.param("~~~text\n## Setup\n~~~", id="a-tilde-fence"),
        pytest.param("    ## Setup", id="an-indented-code-block"),
    ],
)
def test_a_heading_inside_code_is_no_section(tmp_path: Path, block: str) -> None:
    """CommonMark reads all three as code, so the guide's outline does too."""
    overview = "Sorts records by key.\n"
    assert overview in GOOD
    text = GOOD.replace(overview, f"{overview}\n{block}\n", 1)

    assert _rules_broken(tmp_path, text) == []


def test_a_heading_s_closing_hashes_are_not_its_text(tmp_path: Path) -> None:
    assert "## Setup\n" in GOOD

    assert _rules_broken(tmp_path, GOOD.replace("## Setup\n", "## Setup ##\n", 1)) == []


@pytest.mark.parametrize(
    ("row", "written", "detail"),
    [
        pytest.param(
            "| **Devices** | CPU-compatible |",
            "| **Devices** | GPU |",
            "Devices is 'GPU'; one of ['CPU-compatible', 'GPU-optional', 'GPU-required']",
            id="devices",
        ),
        pytest.param(
            "| **Tier** | Quick Reference |",
            "| **Tier** | Intro |",
            "Tier is 'Intro'; one of ['Quick Reference', 'Tutorial', 'Advanced Guide']",
            id="tier",
        ),
    ],
)
def test_a_field_value_outside_the_guide_is_reported_with_the_values_it_may_take(
    tmp_path: Path, row: str, written: str, detail: str
) -> None:
    assert row in GOOD

    findings = _findings(tmp_path, GOOD.replace(row, written, 1))

    assert [finding.detail for finding in findings if finding.rule == "metadata"] == [detail]
