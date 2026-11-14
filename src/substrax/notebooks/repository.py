"""The git queries the example tools make about a repository.

Git runs by its absolute path and without the caller's ``GIT_*`` variables: inside a commit hook
they name that commit's index, not the repository asked about. The queries need git 2.30 or later,
the release whose ``rev-parse`` takes ``--end-of-options``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class RepositoryError(RuntimeError):
    """Git is not installed, or the directory is not in a git repository."""


def unclean_files(repo_root: Path) -> set[str]:  # noqa: DOC502  # RepositoryError is raised by _git
    """The files whose working copy differs from the index, or that git does not track.

    Each file is named on its own, including every file inside an untracked directory. What the
    repository ignores is left out.

    Args:
        repo_root: The repository root.

    Returns:
        The paths, relative to the root: modified or removed tracked files, and untracked files.

    Raises:
        RepositoryError: If git is not installed or ``repo_root`` is not in a git repository.
    """
    listed = _git(repo_root, "ls-files", "-z", "--modified", "--others", "--exclude-standard")
    return set(listed.split("\0")) - {""}


def changed_since(repo_root: Path, ref: str) -> set[str]:  # noqa: DOC502  # RepositoryError is raised by _merge_base and _git
    """The paths below the root changed since ``ref``'s and HEAD's merge base, or new.

    The comparison starts where HEAD's history left ``ref``'s, as a pull request's diff does
    (``git diff ref...HEAD``), so what ``ref`` gained after that point is not a change here; it
    ends at the working tree, so staged and unstaged edits count. In a repository whose git top
    level lies above it, only the paths below the root count, named from the root, as every other
    path the example tools hold is.

    Args:
        repo_root: The repository root.
        ref: A commit, branch or tag; read as a revision even when it looks like an option.

    Returns:
        Every path changed, added (staged or untracked) or removed since the merge base; what the
        repository ignores is left out.

    Raises:
        RepositoryError: If git is not installed, ``repo_root`` is not in a git repository,
            ``ref`` names no commit, or ``ref`` and HEAD share no history.
    """
    base = _merge_base(repo_root, ref)
    changed = _git(repo_root, "diff", "--name-only", "--relative", "-z", base, "--")
    untracked = _git(repo_root, "ls-files", "-z", "--others", "--exclude-standard")
    return (set(changed.split("\0")) | set(untracked.split("\0"))) - {""}


def _merge_base(repo_root: Path, ref: str) -> str:
    """The commit where HEAD's history left ``ref``'s.

    Args:
        repo_root: The repository root.
        ref: A commit, branch or tag.

    Returns:
        The merge base's full hash.

    Raises:
        RepositoryError: If ``ref`` names no commit, or it and HEAD share no history.
    """
    commit = _commit(repo_root, ref)
    base = _git(repo_root, "merge-base", commit, "HEAD", allowed=(0, 1)).strip()
    if not base:
        msg = f"{ref!r} and HEAD have no history in common in {repo_root}"
        raise RepositoryError(msg)
    return base


def _commit(repo_root: Path, ref: str) -> str:
    """The commit ``ref`` names; ``--end-of-options`` keeps git from reading it as an option.

    Args:
        repo_root: The repository root.
        ref: A commit, branch or tag.

    Returns:
        The commit's full hash.

    Raises:
        RepositoryError: If ``ref`` names no commit, as a missing ref, a tree or an option does.
    """
    commit = _git(
        repo_root,
        "rev-parse",
        "--verify",
        "--quiet",
        "--end-of-options",
        f"{ref}^{{commit}}",
        allowed=(0, 1),
    ).strip()
    if not commit:
        msg = f"{ref!r} names no commit in {repo_root}"
        raise RepositoryError(msg)
    return commit


def tracked(repo_root: Path, pattern: str) -> list[Path]:  # noqa: DOC502  # RepositoryError is raised by _git
    """The tracked files matching a git pathspec that the working tree holds, under the root.

    git's index still lists a file deleted and not yet staged; it is left out, since what reads
    these files reads the working tree.

    Args:
        repo_root: The repository root.
        pattern: A git pathspec, such as ``*.md``.

    Returns:
        The files, sorted.

    Raises:
        RepositoryError: If git is not installed or ``repo_root`` is not in a git repository.
    """
    names = _git(repo_root, "ls-files", "-z", "--", pattern).split("\0")
    return sorted(repo_root / name for name in names if name and (repo_root / name).is_file())


def head(repo_root: Path) -> str:  # noqa: DOC502  # RepositoryError is raised by _git
    """The commit HEAD names, or an empty string in a repository with no commit yet.

    Args:
        repo_root: The repository root.

    Returns:
        The commit's full hash, or ``""``.

    Raises:
        RepositoryError: If git is not installed or ``repo_root`` is not in a git repository.
    """
    return _git(repo_root, "rev-parse", "--verify", "--quiet", "HEAD", allowed=(0, 1)).strip()


def index_listing(repo_root: Path) -> str:  # noqa: DOC502  # RepositoryError is raised by _git
    """The index as git lists it: each staged path with its mode, blob and stage.

    Args:
        repo_root: The repository root.

    Returns:
        ``git ls-files --stage`` output; it changes when anything is staged or unstaged.

    Raises:
        RepositoryError: If git is not installed or ``repo_root`` is not in a git repository.
    """
    return _git(repo_root, "ls-files", "--stage", "-z")


def hooks_directory(repo_root: Path) -> Path:  # noqa: DOC502  # RepositoryError is raised by _git
    """The directory git runs hooks from, ``core.hooksPath`` respected.

    Args:
        repo_root: The repository root.

    Returns:
        The hooks directory, which may not exist.

    Raises:
        RepositoryError: If git is not installed or ``repo_root`` is not in a git repository.
    """
    return repo_root / _git(repo_root, "rev-parse", "--git-path", "hooks").strip()


def _git(repo_root: Path, *arguments: str, allowed: tuple[int, ...] = (0,)) -> str:
    """Run one git command in the repository and return its output.

    Args:
        repo_root: The directory git runs in.
        *arguments: The git command and its arguments.
        allowed: The exit codes that are not a failure.

    Returns:
        What the command printed.

    Raises:
        RepositoryError: If git is not installed or the command fails, as it does outside a git
            repository.
    """
    git = shutil.which("git")
    if git is None:
        msg = "git is not installed"
        raise RepositoryError(msg)
    completed = subprocess.run(  # noqa: S603  # nosec B603 - absolute path from shutil.which
        [git, *arguments],
        cwd=repo_root,
        env={key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in allowed:
        msg = f"git {arguments[0]} failed in {repo_root}: {completed.stderr.strip()}"
        raise RepositoryError(msg)
    return completed.stdout
