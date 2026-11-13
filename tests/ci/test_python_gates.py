"""The shared ``python-gates`` workflow every stack repository calls, defined once.

It runs the Required gates of the engineering guide: the lock check, ruff, pyright, pre-commit,
pytest with the coverage floor and the changed-line floor, the dependency audit with Bandit, the
build with ``twine check``, a clean-install import, and the strict docs build. Repositories call it
at the job level, pinned by commit, so every rule here holds for every caller.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import venv
from pathlib import Path
from typing import Any

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "python-gates.yml"
INPUTS = {
    "package",
    "python-versions",
    "sync-args",
    "docs-sync-args",
    "pytest-args",
    "test-env",
    "import-modules",
    "os",
    "already-tested",
    "only-security",
    "uv-version",
    "test-shards",
}
JOBS = {"lint", "test", "coverage", "security", "build", "docs"}
# Two runners at a time on macOS, scarce for every repository; elsewhere 256, the most jobs a
# matrix may make, which is no cap.
MACOS_CAP = "${{ startsWith(inputs.os, 'macos') && 2 || 256 }}"
# The stack is a hierarchy built bottom-up from substrax: its shared workflow may use substrax's
# own actions and third-party actions, never another stack repository's.
STACK_OWNERS = ("avitai/", "mahdi-shafiei/")


def _workflow() -> dict[str, Any]:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)  # noqa: S506


def _jobs() -> dict[str, Any]:
    return _workflow()["jobs"]


def _steps(job: str) -> list[dict[str, Any]]:
    return list(_jobs()[job]["steps"])


def _runs(job: str) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(job))


def _env(step: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in (step.get("env") or {}).items()}


def _uses() -> list[str]:
    return [
        str(step["uses"])
        for job in _jobs().values()
        for step in job.get("steps", [])
        if "uses" in step
    ]


def test_it_runs_only_when_called_with_the_declared_inputs() -> None:
    workflow = _workflow()

    assert set(workflow["on"]) == {"workflow_call"}
    assert set(workflow["on"]["workflow_call"]["inputs"]) == INPUTS
    assert workflow["on"]["workflow_call"]["inputs"]["package"]["required"] == "true"


def test_it_leaves_concurrency_and_env_to_the_caller() -> None:
    """A called workflow's `github.workflow` is the caller's.

    So a group here would cancel the caller.
    """
    workflow = _workflow()

    assert "concurrency" not in workflow
    assert "env" not in workflow
    assert [name for name, job in _jobs().items() if "concurrency" in job] == []


def test_it_runs_the_gate_jobs() -> None:
    assert set(_jobs()) == JOBS


def test_lint_runs_the_lock_check_pyright_and_every_hook() -> None:
    """pyright runs here itself: a repository's pre-commit file need not hold a pyright hook.

    ruff runs through the ruff and ruff-format hooks every stack repository's pre-commit file holds.
    A pathless `ruff format --check` would also format Python blocks inside Markdown, a gate no
    repository runs today.
    """
    runs = _runs("lint")

    for command in ("uv lock --check", "pyright --warnings", "pre-commit run --all-files"):
        assert command in runs, command


def test_pre_commit_leaves_the_suite_bandit_and_pyright_to_the_jobs_that_run_them() -> None:
    step = next(step for step in _steps("lint") if "pre-commit run" in str(step.get("run", "")))

    skipped = set(_env(step)["SKIP"].split(","))
    assert {"pytest", "bandit", "pyright"} <= skipped
    assert not {"ruff", "ruff-format"} & skipped


def test_lint_is_never_stood_down_for_an_already_tested_tree() -> None:
    """Lint saves the pre-commit cache main must hold, so a merge still runs it."""
    condition = _jobs()["lint"].get("if", "")

    assert "already-tested" not in condition


def test_lint_restores_the_hook_environments_before_running_the_hooks() -> None:
    """Hook environments are built once per configuration, runner and Python, not every run.

    Without the cache every run installs every hook's environment again; the ``shellcheck-py``
    hook fetches the ShellCheck binary from GitHub releases, and an HTTP 504 there failed a run.
    """
    steps = _steps("lint")
    cache = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("actions/cache@")
        and step["with"]["path"] == "~/.cache/pre-commit"
    )
    pre_commit = next(
        i for i, step in enumerate(steps) if "pre-commit run" in str(step.get("run", ""))
    )
    key = steps[cache]["with"]["key"]

    assert cache < pre_commit
    assert "hashFiles('.pre-commit-config.yaml')" in key
    assert "runner.os" in key
    assert "inputs.python-versions" in key


@pytest.mark.parametrize("job", ["test", "coverage", "security", "build", "docs"])
def test_work_a_pull_request_already_proved_stands_down(job: str) -> None:
    assert "inputs.already-tested != 'true'" in _jobs()[job]["if"]


@pytest.mark.parametrize("job", ["lint", "test", "coverage", "build", "docs"])
def test_the_scheduled_audit_runs_security_alone(job: str) -> None:
    assert "!inputs.only-security" in _jobs()[job]["if"]
    assert "only-security" not in _jobs()["security"].get("if", "")


@pytest.mark.parametrize(("job", "cap"), [("test", MACOS_CAP), ("build", "2")])
def test_the_matrix_jobs_run_on_the_callers_runner_under_the_macos_cap(job: str, cap: str) -> None:
    """The build has two legs; the tests' shards run all at once except on macOS."""
    spec = _jobs()[job]

    assert spec["runs-on"] == "${{ inputs.os }}"
    assert spec["strategy"]["max-parallel"] == cap
    assert spec["strategy"]["fail-fast"] == "false"
    assert spec["strategy"]["matrix"]["python-version"] == "${{ fromJSON(inputs.python-versions) }}"


def _pytest_step() -> dict[str, Any]:
    return next(step for step in _steps("test") if "pytest" in str(step.get("run", "")))


def test_the_tests_run_in_the_callers_shards() -> None:
    """One job per Python and shard; a caller that names no shards runs the suite whole."""
    spec = _jobs()["test"]
    inputs = _workflow()["on"]["workflow_call"]["inputs"]

    assert json.loads(inputs["test-shards"]["default"]) == [1]
    assert spec["strategy"]["matrix"]["group"] == "${{ fromJSON(inputs.test-shards) }}"
    assert "${{ matrix.group }}" in spec["name"]
    assert (
        _env(_pytest_step()).items()
        >= {
            "SHARDS": "${{ inputs.test-shards }}",
            "GROUP": "${{ matrix.group }}",
            "SEED": "${{ github.run_id }}",
        }.items()
    )


def _pytest_arguments(tmp_path: Path, *, shards: str, group: str) -> list[str]:
    """The arguments the test step gives pytest, its ``uv`` stubbed to print them one a line."""
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "uv").write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    (shim / "uv").chmod(0o755)
    env = {
        "PATH": os.pathsep.join([str(shim), os.environ.get("PATH", "")]),
        "PYTEST_ARGS": "tests/ -p randomly -v",
        "SHARDS": shards,
        "GROUP": group,
        "SEED": "4242",
    }
    bash = [shutil.which("bash") or "bash", "--noprofile", "--norc", "-eo", "pipefail", "-c"]
    result = subprocess.run(  # noqa: S603  # nosec B603 - bash running the workflow's own step
        [*bash, str(_pytest_step()["run"])],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def test_a_shard_runs_its_group_of_one_collection_ordered_by_the_runs_seed(tmp_path: Path) -> None:
    """pytest-randomly orders the collection before pytest-split assigns it, so every shard of a run
    takes the run's id as its seed: one order, each test in exactly one shard. Separate seeds
    duplicated 110 tests and dropped 119 in datarax (OWN-1001-SPEED)."""
    arguments = _pytest_arguments(tmp_path, shards="[1, 2, 3, 4]", group="3")

    assert arguments[:3] == ["run", "--no-sync", "pytest"]
    assert arguments[3:7] == ["tests/", "-p", "randomly", "-v"]
    assert " ".join(arguments[7:]).split(" --cov")[0] == (
        "--splits 4 --group 3 --splitting-algorithm least_duration --randomly-seed=4242"
    )


def test_an_unsharded_run_passes_no_split_or_seed(tmp_path: Path) -> None:
    """A caller with one shard needs neither pytest-split nor pytest-randomly installed."""
    arguments = _pytest_arguments(tmp_path, shards="[1]", group="1")

    assert not [argument for argument in arguments if argument.startswith(("--splits", "--group"))]
    assert not [argument for argument in arguments if argument.startswith("--randomly-seed")]


def test_each_shard_reports_its_coverage_for_the_coverage_job_to_hold(tmp_path: Path) -> None:
    """A shard measures part of the suite, so its own floor is off and its data goes on."""
    arguments = _pytest_arguments(tmp_path, shards="[1, 2]", group="1")
    upload = next(
        step
        for step in _steps("test")
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    )

    assert "--cov-fail-under=0" in arguments
    assert "--cov-report=xml" in arguments
    assert upload["with"]["name"] == "coverage-${{ matrix.python-version }}-${{ matrix.group }}"
    assert upload["with"]["path"] == ".coverage"
    assert upload["with"]["include-hidden-files"] == "true"
    assert upload["with"]["if-no-files-found"] == "error"


def test_the_coverage_job_holds_each_pythons_shards_together_to_the_floors() -> None:
    """Each Python's shards are combined and reported against the configured floor, and the lines a
    pull request changes are held to 80% on the first Python, as one unsharded leg held them."""
    spec = _jobs()["coverage"]
    download = next(
        step
        for step in _steps("coverage")
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    )
    runs = _runs("coverage")
    diff = next(step for step in _steps("coverage") if "diff-cover" in str(step.get("run", "")))
    checkout = next(
        step
        for step in _steps("coverage")
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )

    assert spec["needs"] == "test"
    assert spec["runs-on"] == "ubuntu-latest"
    assert spec["strategy"]["matrix"]["python-version"] == "${{ fromJSON(inputs.python-versions) }}"
    assert download["with"]["pattern"] == "coverage-${{ matrix.python-version }}-*"
    for command in ("coverage combine", "coverage report", "coverage xml"):
        assert command in runs, command
    assert "--fail-under" not in runs.split("diff-cover")[0], "the floor is the configured one"
    assert "--fail-under=80" in diff["run"]
    assert "--compare-branch=origin/main" in diff["run"]
    assert "github.event_name == 'pull_request'" in diff["if"]
    assert FIRST_PYTHON.removeprefix("${{ ").removesuffix(" }}") in diff["if"]
    assert checkout["with"]["fetch-depth"] == "0"


def test_security_audits_the_lock_after_setup_uv_through_substrax_by_commit() -> None:
    """A called workflow resolves `./` in the caller's checkout, so the action is named in full."""
    uses = [str(step.get("uses", "")) for step in _steps("security")]

    audit = next(index for index, use in enumerate(uses) if "/.github/actions/audit-lock@" in use)
    assert re.fullmatch(r"avitai/substrax/\.github/actions/audit-lock@[0-9a-f]{40}", uses[audit])
    assert any(use.startswith("astral-sh/setup-uv@") for use in uses[:audit])


def test_bandit_runs_whatever_the_audit_found_and_its_failure_fails_the_job() -> None:
    bandit = next(step for step in _steps("security") if "bandit " in str(step.get("run", "")))

    assert bandit.get("if") == "${{ !cancelled() }}"
    assert "continue-on-error" not in bandit
    assert "||" not in bandit["run"]
    assert '"src/$PACKAGE"' in bandit["run"]
    assert _env(bandit)["PACKAGE"] == "${{ inputs.package }}"


def test_bandit_runs_once_as_the_engineering_guides_gate() -> None:
    """The guide's gate is one command, ``bandit -r src/ -ll``, and the job runs only that.

    A second run that only wrote a JSON report stated the threshold again and left it at Bandit's
    default, Low: findings the gate passes failed the step after it. Nothing read the report.
    """
    runs = [line.strip() for line in _runs("security").splitlines() if "bandit " in line]
    uploads = [
        step["with"]["path"]
        for step in _steps("security")
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]

    assert len(runs) == 1, runs
    assert re.search(r"\sbandit -r \"src/\$PACKAGE\" -c pyproject\.toml -ll(\s|$)", runs[0])
    assert uploads == []


def _import_check() -> dict[str, Any]:
    check = next(
        (step for step in _steps("build") if "importlib.metadata" in str(step.get("run", ""))),
        None,
    )
    assert check is not None, "the build job has no step that checks the installed version"
    return check


def test_the_build_is_checked_and_installed_in_a_clean_environment_before_the_import() -> None:
    steps = _steps("build")
    runs = _runs("build")
    install = next(
        i for i, step in enumerate(steps) if "uv pip install" in str(step.get("run", ""))
    )

    assert "uv build" in runs
    assert "twine check --strict" in runs
    assert install < steps.index(_import_check())
    assert _env(_import_check()) == {
        "PACKAGE": "${{ inputs.package }}",
        "MODULES": "${{ inputs.import-modules }}",
    }


def test_the_first_python_leg_uploads_the_distributions_a_release_publishes() -> None:
    """A repository's ``publish.yml`` releases the distributions CI built and checked, by this name.

    One leg uploads, since an artifact name is unique within a run, and only after the import check,
    so a release never carries distributions that failed it.
    """
    steps = _steps("build")
    upload = next(
        (
            step
            for step in steps
            if str(step.get("uses", "")).startswith("actions/upload-artifact@")
        ),
        None,
    )

    assert upload is not None, "the build job uploads no distributions"
    assert steps.index(_import_check()) < steps.index(upload)
    assert upload["if"] == "${{ matrix.python-version == fromJSON(inputs.python-versions)[0] }}"
    assert upload["with"]["name"] == "python-package-distributions"
    assert upload["with"]["path"] == "dist/"
    assert upload["with"]["if-no-files-found"] == "error"


def _run_import_check(
    tmp_path: Path,
    *,
    declared: str | None,
    installed: str = "1.0",
    modules: str = "",
    source_version: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the build job's import check, as the runner does, against a fake installed wheel.

    ``fakepkg`` declares ``declared`` as ``__version__`` (none when None) and is installed as the
    distribution ``fake-dist`` at ``installed``, the name the project's ``pyproject.toml`` gives.
    ``source_version`` puts a ``fakepkg`` source tree with that version in the project directory,
    where a flat-layout repository keeps its package.
    """
    runner = tmp_path / "runner"
    clean = runner / "clean"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(clean)
    purelib = Path(
        subprocess.run(  # noqa: S603  # nosec B603 - the venv's python with fixed arguments
            [
                clean / "bin" / "python",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    (purelib / "fakepkg").mkdir()
    (purelib / "fakepkg" / "__init__.py").write_text(
        "" if declared is None else f'__version__ = "{declared}"\n', encoding="utf-8"
    )
    metadata = purelib / f"fake_dist-{installed}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: fake-dist\nVersion: {installed}\n", encoding="utf-8"
    )

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "fake-dist"\n', encoding="utf-8")
    if source_version is not None:
        (project / "fakepkg").mkdir()
        (project / "fakepkg" / "__init__.py").write_text(
            f'__version__ = "{source_version}"\n', encoding="utf-8"
        )

    # The runner's default shell for a `run:` step.
    bash = [shutil.which("bash") or "bash", "--noprofile", "--norc", "-eo", "pipefail", "-c"]
    return subprocess.run(  # noqa: S603  # nosec B603 - bash running the workflow's own step
        [*bash, str(_import_check()["run"])],
        cwd=project,
        env={**os.environ, "RUNNER_TEMP": str(runner), "PACKAGE": "fakepkg", "MODULES": modules},
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_import_check_passes_a_version_that_matches_the_installed_metadata(
    tmp_path: Path,
) -> None:
    result = _run_import_check(tmp_path, declared="1.0")

    assert result.returncode == 0, result.stderr


def test_the_import_check_fails_a_version_the_installed_metadata_contradicts(
    tmp_path: Path,
) -> None:
    result = _run_import_check(tmp_path, declared="0.9", installed="1.0")

    assert result.returncode != 0
    assert "0.9" in result.stderr
    assert "1.0" in result.stderr
    assert "fake-dist" in result.stderr


def test_the_import_check_passes_a_package_that_declares_no_version(tmp_path: Path) -> None:
    """Four stack packages declare no ``__version__``; the metadata is then the only version."""
    result = _run_import_check(tmp_path, declared=None)

    assert result.returncode == 0, result.stderr


def test_the_import_check_fails_a_listed_module_that_does_not_import(tmp_path: Path) -> None:
    result = _run_import_check(tmp_path, declared="1.0", modules="fakepkg fakepkg.missing")

    assert result.returncode != 0
    assert "fakepkg.missing" in result.stderr


def test_the_import_check_reads_the_installed_wheel_not_the_source_tree(tmp_path: Path) -> None:
    """The check runs in the repository's root.

    There, a flat-layout package would shadow the wheel.
    """
    result = _run_import_check(tmp_path, declared="1.0", source_version="9.9")

    assert result.returncode == 0, result.stderr


# The jobs that install the project's environment, and the Python each installs it on.
FIRST_PYTHON = "${{ fromJSON(inputs.python-versions)[0] }}"
ENVIRONMENT_PYTHON = {
    "lint": FIRST_PYTHON,
    "test": "${{ matrix.python-version }}",
    "coverage": "${{ matrix.python-version }}",
    "security": FIRST_PYTHON,
    "docs": FIRST_PYTHON,
}


def _install_step(job: str) -> dict[str, Any]:
    return next(step for step in _steps(job) if "uv sync --locked" in str(step.get("run", "")))


@pytest.mark.parametrize(("job", "python"), sorted(ENVIRONMENT_PYTHON.items()))
def test_each_environment_job_names_its_python_once_in_uv_python(job: str, python: str) -> None:
    """``UV_PYTHON`` on the job is the only place its Python is named.

    A caller's tracked ``.python-version`` is the interpreter request of every uv project command,
    and ``uv sync`` replaces a venv made on another version, so ``uv venv --python`` alone is
    undone by the sync after it. ``UV_PYTHON`` applies to every uv command in the job.
    """
    install = _install_step(job)

    assert _jobs()[job]["env"]["UV_PYTHON"] == python
    assert "PYTHON" not in _env(install)
    assert "--python" not in install["run"]


def _running_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _resolve(value: str) -> str:
    """The value a job's ``env`` entry takes on a runner testing the running Python."""
    known = {
        "${{ matrix.python-version }}": _running_python(),
        FIRST_PYTHON: _running_python(),
        "${{ inputs.sync-args }}": "",
        "${{ inputs.docs-sync-args }}": "",
    }
    if "${{" not in value:
        return value
    assert value in known, f"no runner value for {value}"
    return known[value]


def _run_install_step(
    tmp_path: Path, job: str, *, pinned: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run ``job``'s install step, as the runner does, in a project pinning ``pinned``.

    The project has no dependencies, so uv works offline against the running interpreter. A
    ``uv`` shim first on ``PATH`` stubs the two calls a runner makes over the network or with
    the docs toolchain: ``uv python install`` and the docs job's ``mkdocs build``. Every other
    call reaches the real uv.
    """
    real_uv = shutil.which("uv")
    assert real_uv is not None, "the gates need uv on PATH"
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "0"\nrequires-python = ">=3.10"\n',
        encoding="utf-8",
    )
    (project / ".python-version").write_text(f"{pinned}\n", encoding="utf-8")
    (project / "mkdocs.yml").write_text("site_name: fixture\n", encoding="utf-8")
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "uv").write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1 $2" = "python install" ]; then exit 0; fi\n'
        'if [ "$1 $2 $3" = "run --no-sync mkdocs" ]; then exit 0; fi\n'
        f'exec "{real_uv}" "$@"\n',
        encoding="utf-8",
    )
    (shim / "uv").chmod(0o755)
    # The interpreter under the test's venv, first on PATH so uv finds the running version.
    base = Path(sys.base_prefix) / "bin"
    # The runner's own uv settings stay out: substrax's test job sets UV_PYTHON, and an inherited
    # one would hold the venv's Python whatever the step under test does.
    env = {key: value for key, value in os.environ.items() if not key.startswith("UV_")}
    env.pop("VIRTUAL_ENV", None)
    env.update(
        PATH=os.pathsep.join([str(shim), str(base), env.get("PATH", "")]),
        UV_CACHE_DIR=str(tmp_path / "cache"),
        UV_OFFLINE="1",
        UV_PYTHON_DOWNLOADS="never",
    )
    env.update({key: _resolve(value) for key, value in (_jobs()[job].get("env") or {}).items()})
    step = _install_step(job)
    env.update({key: _resolve(value) for key, value in _env(step).items()})
    locked = subprocess.run(  # noqa: S603  # nosec B603 - uv with fixed arguments
        [real_uv, "lock", "--python", _running_python()],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert locked.returncode == 0, locked.stderr
    bash = [shutil.which("bash") or "bash", "--noprofile", "--norc", "-eo", "pipefail", "-c"]
    result = subprocess.run(  # noqa: S603  # nosec B603 - bash running the workflow's own step
        [*bash, str(step["run"])],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, project / ".venv" / "bin" / "python"


@pytest.mark.parametrize("job", sorted(ENVIRONMENT_PYTHON))
def test_each_environment_job_installs_on_its_python_whatever_python_version_says(
    tmp_path: Path, job: str
) -> None:
    """A caller's ``.python-version`` naming another Python does not move the job's venv.

    The fixture pins 3.99, which no runner has: a step whose ``uv sync`` follows the file fails
    there, and one that holds the job's Python builds its venv on the running interpreter.
    """
    result, python = _run_install_step(tmp_path, job, pinned="3.99")

    assert result.returncode == 0, result.stderr
    version = subprocess.run(  # noqa: S603  # nosec B603 - the venv's python with fixed arguments
        [python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert version == _running_python()


def test_the_docs_build_strictly_with_no_path_filter() -> None:
    docs = next(step for step in _steps("docs") if "mkdocs build" in str(step.get("run", "")))

    assert "mkdocs build --strict" in docs["run"]
    assert _env(docs)["DOCS_SYNC_ARGS"] == "${{ inputs.docs-sync-args }}"


def test_no_script_interpolates_an_input() -> None:
    """An input reaches a script through `env`.

    Never as `${{ }}` text the shell would run as code.
    """
    interpolated = [
        f"{name}: {step.get('name', step['run'][:40])}"
        for name, job in _jobs().items()
        for step in job.get("steps", [])
        if "run" in step and "${{" in str(step["run"])
    ]

    assert interpolated == []


def test_it_uses_no_stack_repository_but_substrax() -> None:
    others = [
        use
        for use in _uses()
        if use.startswith(STACK_OWNERS) and not use.startswith("avitai/substrax/")
    ]

    assert others == []


def test_every_setup_uv_installs_the_pinned_uv_and_prunes_its_cache() -> None:
    inputs = _workflow()["on"]["workflow_call"]["inputs"]
    setups = [
        step["with"]
        for job in _jobs().values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]

    assert re.fullmatch(r"\d+\.\d+\.\d+", inputs["uv-version"]["default"])
    assert len(setups) == len(JOBS)
    for spec in setups:
        assert spec["version"] == "${{ inputs.uv-version }}"
        assert spec["prune-cache"] == "true"
