"""The shared example guide states the rules the code checks, and its template follows them."""

from __future__ import annotations

import os
import re
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path

import jupytext
from _example_texts import CONFIG, write_example

from substrax.notebooks.conformance import (
    check_example,
    Devices,
    RULES,
    SECTIONS,
    SOURCE_RULES,
    Tier,
)
from substrax.notebooks.run import ExampleRun, execute_example


REPOSITORY = Path(__file__).resolve().parents[2]
GUIDES = REPOSITORY / "docs" / "guides"
GUIDE = (GUIDES / "writing-examples.md").read_text(encoding="utf-8")
SUITE_CHECKS = {"corpus", "runs-as-its-page-shows"}
"""The shared example tests' own checks, beside the rules of substrax.notebooks.conformance."""


def _checked(text: str) -> list[str]:
    return re.findall(r"Checked:\s+`([a-z-]+)`", text)


def test_every_checked_tag_in_the_guide_names_a_check_and_every_check_is_in_the_guide() -> None:
    tags = _checked(GUIDE)

    assert sorted(set(tags)) == sorted(set(RULES) | SUITE_CHECKS)
    assert len(tags) == len(set(tags)), "a check is tagged in two places"


def test_the_guide_names_every_tier_device_and_section_the_checks_accept() -> None:
    names = [*Tier, *Devices, *(f"## {section.rstrip('?')}" for section in SECTIONS)]

    assert [name for name in names if f"`{name}`" not in GUIDE] == []


def test_every_command_the_guide_gives_is_one_substrax_installs() -> None:
    scripts = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    installed = set(scripts["project"]["scripts"])
    named = set(re.findall(r"uv run (substrax-[a-z]+)", GUIDE))

    assert named
    assert named <= installed


def _gpu_lane() -> list[str]:
    """The commands of the guide's GPU lane, one per line: its shell block checking for a GPU."""
    blocks = re.findall(r"```bash\n(.*?)```", GUIDE, flags=re.DOTALL)
    (lane,) = [block for block in blocks if "--require-gpu" in block]
    return [line.strip() for line in lane.strip().splitlines()]


def test_the_guide_s_gpu_lane_fails_before_the_examples_where_the_default_backend_is_no_gpu() -> (
    None
):
    """Its check is ``require_gpu``'s, which reads only jax's default backend.

    The test makes the CPU the default with ``JAX_PLATFORMS=cpu`` on every machine. A machine
    without a usable GPU reaches the same branch whatever put the CPU first (no NVIDIA device
    node, a CUDA that cannot start, no CUDA plugin), so the check fails it as it fails this.
    """
    check, examples = _gpu_lane()
    assert examples == "uv run substrax-notebooks test"
    assert "JAX_PLATFORMS" not in check + examples
    program, *arguments = check.removeprefix("uv run ").split()

    result = subprocess.run(  # noqa: S603  # nosec B603 - the guide's command, this venv, no shell
        [str(Path(sys.executable).with_name(program)), *arguments],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "no GPU: the default backend is cpu" in result.stderr


def test_the_template_follows_every_source_rule(tmp_path: Path) -> None:
    text = (GUIDES / "example_template.py").read_text(encoding="utf-8")
    repo, script = write_example(tmp_path, text, relative="statistics/normalising.py")

    assert [
        str(finding)
        for finding in check_example(script, repo_root=repo, config=CONFIG, rules=SOURCE_RULES)
    ] == []


RESULT_TEST = GUIDES / "example_result_test.py"
"""The guide's test of the template's results, a file the repository's ruff and pyright check."""


def test_the_guide_quotes_its_result_test_as_the_repository_checks_it() -> None:
    """The sample passes the guide's own rule: the repository's ruff and pyright (T-5)."""
    assert f"```python\n{RESULT_TEST.read_text(encoding='utf-8')}```" in GUIDE


def test_the_guide_s_result_test_passes_against_the_template_s_run(tmp_path: Path) -> None:
    text = (GUIDES / "example_template.py").read_text(encoding="utf-8")
    repo, script = write_example(tmp_path, text, relative="statistics/normalising.py")
    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    def example_run(path: str) -> ExampleRun:
        assert path == "statistics/normalising.py"
        return run

    sample = runpy.run_path(str(RESULT_TEST))
    tests = [value for name, value in sample.items() if name.startswith("test_")]
    assert len(tests) == 1
    tests[0](example_run)


PAGE_REWRITING_HOOKS = ("markdownlint-fix", "trailing-whitespace", "detect-secrets")
"""Hooks measured rewriting or rejecting a generated page: markdownlint-fix reformats nbconvert's
indented output blocks, trailing-whitespace strips an output's trailing spaces (the page then reads
as edited), and detect-secrets 1.5.0 flags the provenance record's digests."""


def test_the_guide_excludes_generated_pages_from_every_hook_measured_to_change_them() -> None:
    exclusion = GUIDE.split("Generated pages are left out of", 1)[1].split("\n\n```", 1)[0]

    assert [hook for hook in PAGE_REWRITING_HOOKS if hook not in exclusion] == []


def test_the_template_s_run_is_shown_under_results_and_next_steps_closes_it() -> None:
    """The page shows a cell's output beneath it, so ``main()``'s cell sits under Results (K4)."""
    cells = jupytext.reads(
        (GUIDES / "example_template.py").read_text(encoding="utf-8"), fmt="py:percent"
    ).cells
    first = {
        heading: next(i for i, cell in enumerate(cells) if f"\n{heading}\n" in f"\n{cell.source}\n")
        for heading in ("## Results", "## Troubleshooting", "## Next Steps")
    }
    main = next(i for i, cell in enumerate(cells) if "def main(" in cell.source)

    assert first["## Results"] < main < first["## Troubleshooting"] <= first["## Next Steps"]
    assert [i for i, cell in enumerate(cells) if cell.cell_type == "code"][-1] == main


def _fenced(language: str) -> list[str]:
    """Every block of ``language`` the guide fences."""
    return re.findall(rf"```{language}\n(.*?)```", GUIDE, flags=re.DOTALL)


def test_the_guide_leaves_every_notebook_to_convert_even_when_a_hook_names_it() -> None:
    """``ruff check --fix`` on a notebook left the pair out-of-sync; ``extend-exclude`` alone did
    not stop a hook that passes the notebook's path, ``force-exclude`` did (ruff 0.16.6, K3)."""
    ruff = [
        tomllib.loads(block)["tool"]["ruff"] for block in _fenced("toml") if "[tool.ruff]" in block
    ]

    assert len(ruff) == 1
    assert "examples/**/*.ipynb" in ruff[0]["extend-exclude"]
    assert ruff[0]["force-exclude"] is True


def test_the_guide_fetches_the_base_its_pull_request_command_diffs_against() -> None:
    """A one-commit checkout has no ``origin/main``, which the command refuses (T-7)."""
    workflow = [block for block in _fenced("yaml") if "--changed-since origin/main" in block]

    assert len(workflow) == 1
    assert "fetch-depth: 0" in workflow[0]


def test_the_guide_says_what_a_next_steps_link_is_relative_to() -> None:
    """The same text sits in the script, the notebook and the page; only the page's base is
    checked (``mkdocs build --strict`` fails on a page-relative link naming no page, T-19)."""
    section = GUIDE.split("8. `## Next Steps`", 1)[1].split("\n## ", 1)[0]
    next_steps = " ".join(section.split())

    assert "relative to the example's page" in next_steps
    assert "mkdocs build --strict" in next_steps
