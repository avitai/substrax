# Contributing

Thanks for helping improve Substrax. This repository is still early, so keep
changes small, tested, and explicit about user-visible behavior.

## Local Setup

```bash
git clone https://github.com/avitai/substrax.git
cd substrax
./setup.sh
source activate.sh
uv run pre-commit install
```

Run all project commands after activating the local environment:

```bash
source activate.sh
uv run --locked pytest
uv run --locked pre-commit run --all-files
uv run --locked mkdocs build --strict --clean
```

## Contribution Workflow

1. Create a focused branch from `main`.
2. Add or update tests before changing behavior.
3. Keep docs and README claims aligned with the current code.
4. Run the relevant targeted checks plus the full verification stack before
   opening a pull request.
5. Use the pull request checklist and call out any intentionally skipped checks.

## Pull Request Expectations

- Functional changes include tests.
- Behaviour that replaces a copy in a sibling package lands with the tests that
  copy carried, and the copy is deleted in the same change.
- Documentation changes build with strict MkDocs.
- Dependency or workflow changes explain the maintenance impact.

## Reporting Security Issues

Do not open a public issue for suspected vulnerabilities. Follow
[SECURITY.md](SECURITY.md) instead.
