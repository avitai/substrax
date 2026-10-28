# Releasing Substrax

Substrax publishes through `.github/workflows/publish.yml`.
No commit or tag push creates a release by itself. Release timing and versioning
stay under operator control. The manual `target=github-release` workflow path
creates a GitHub Release for an explicit existing tag with
`softprops/action-gh-release@v3` and `generate_release_notes: true`, then
publishes to PyPI with trusted publishing. Publishing an existing GitHub Release
also runs the PyPI upload path.

## Release Checklist

1. Activate the local environment.

   ```bash
   source activate.sh
   ```

2. Bump `version` in `pyproject.toml` (the one source of the version; `substrax.__version__`
   reads it from the installed package).
3. Update `CHANGELOG.md` by moving unreleased entries under the new version and
   date.
4. Run the release checks.

   ```bash
   uv run pytest
   uv run pre-commit run --all-files
   uv run mkdocs build --strict --clean
   rm -rf dist/
   uv build
   uv run twine check dist/*
   ```

5. Commit the version and changelog updates on a branch and open a pull request. `main` accepts
   only a pull request whose `required` check passed, merged by rebase:

   ```bash
   gh pr create --base main --fill
   gh pr merge --rebase   # once `required` has passed
   ```

6. Run the macOS workflow on the release commit and wait for it to pass. No push or pull request
   runs on macOS, so this is the release's macOS check.

   ```bash
   gh workflow run macos.yml --ref main
   gh run watch "$(gh run list --workflow macos.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
   ```

7. Create and push an annotated tag on the release commit `main` holds. A rebase merge gives
   the release commit a new SHA, so the tag is made on `main` after a fast-forward pull, and only
   the tag is pushed: `main` itself changes only through pull requests.

   ```bash
   git switch main
   git pull --ff-only
   target_sha=$(git rev-parse HEAD)
   git tag -a vX.Y.Z -m "substrax X.Y.Z"
   git push origin vX.Y.Z
   ```

8. In GitHub Actions, manually run `Publish to PyPI` with:

   - `target=github-release`
   - `version_tag=vX.Y.Z`

   The workflow verifies that the tag exists, creates the GitHub Release with
   generated release notes, then publishes to PyPI.

## Manual Release Recovery

If the manual generated-release workflow is interrupted before creating the
GitHub Release, use GitHub generated notes manually from the exact tagged
commit.

   ```bash
   gh release create vX.Y.Z --target "$target_sha" --generate-notes
   ```

Publishing that release triggers the same PyPI upload workflow.

## TestPyPI

Use the manual `workflow_dispatch` path in `publish.yml` with
`target=testpypi` when validating trusted publishing setup before a real
release.

## PyPI Trusted Publishing

PyPI must trust:

- Owner: `avitai`
- Repository: `substrax`
- Workflow: `publish.yml`
- Environment: `pypi`

If PyPI rejects the publish with `invalid-publisher`, verify the trusted
publisher registration before looking for repository secrets. The expected
publisher identity is:

```text
repo:avitai/substrax:environment:pypi
```

For TestPyPI, the expected environment is `testpypi`.
