"""The shared example guide states the rules the code checks, and its template follows them."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from _example_texts import CONFIG, write_example

from substrax.notebooks.conformance import (
    check_example,
    Devices,
    RULES,
    SECTIONS,
    SOURCE_RULES,
    Tier,
)
from substrax.notebooks.run import execute_example


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


def test_the_template_runs_and_reports_its_results(tmp_path: Path) -> None:
    text = (GUIDES / "example_template.py").read_text(encoding="utf-8")
    repo, script = write_example(tmp_path, text, relative="statistics/normalising.py")

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert isinstance(run.returned, dict)
    assert run.returned["count"] == 128


PAGE_REWRITING_HOOKS = ("markdownlint-fix", "trailing-whitespace", "detect-secrets")
"""Hooks measured rewriting or rejecting a generated page: markdownlint-fix reformats nbconvert's
indented output blocks, trailing-whitespace strips an output's trailing spaces (the page then reads
as edited), and detect-secrets 1.5.0 flags the provenance record's digests."""


def test_the_guide_excludes_generated_pages_from_every_hook_measured_to_change_them() -> None:
    exclusion = GUIDE.split("Generated pages are left out of", 1)[1].split("\n\n```", 1)[0]

    assert [hook for hook in PAGE_REWRITING_HOOKS if hook not in exclusion] == []
