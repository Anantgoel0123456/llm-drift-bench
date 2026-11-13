# Contributing

Start with a reproducible problem, a small fix, or a documentation improvement.
For larger features, open an issue describing the user problem before coding.
See the [roadmap](docs/roadmap.md) for priorities. Roadmap items are proposals,
not necessarily existing or assigned GitHub issues.

## Local setup, PowerShell

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m pytest --cov=llm_drift_bench --cov-report=term-missing
& .\.venv\Scripts\python.exe -m ruff check src tests
& .\.venv\Scripts\python.exe -m mypy src/llm_drift_bench
& .\.venv\Scripts\python.exe -m build
& .\.venv\Scripts\python.exe -m twine check dist/*
```

On Linux/macOS use `.venv/bin/python`. No activation is required.
Run pytest, not only unittest: some regression tests are pytest functions.

## A useful pull request

- Explain the concrete trigger and the behavior before and after your change.
- Include a small regression test for a bug or feature.
- Keep runtime dependencies justified and Python 3.10 compatibility intact.
- Preserve missing measurements rather than guessing replacements.
- Update documentation and the unreleased changelog for user-visible changes.
- Use separate commits for independent changes, with descriptive messages.

Good first contributions include clearer input-error messages, documentation
reproductions, and narrow loader edge cases. Schema semantics and scoring changes
need careful review and should not be treated as trivial starter tasks.

Never include API keys, private prompts, personal information, or proprietary
model outputs in issues and fixtures. Treat contributors respectfully and review
code on its merits.
