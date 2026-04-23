# optimalcontrol

Python package for NMR spin dynamics implementing analytical ROPE/CROP theories and
numerical GRAPE optimisation, with a Spinach-compatible API, ensemble support, and
paper-reproduction examples.

## Source references

- **JMR 2003 (ROPE)**: Unterbeck & Glaser, *Journal of Magnetic Resonance* 160 (2003) 88–101 — analytical optimal control for heteronuclear transfer under relaxation.
- **PNAS 2003 (CROP)**: Unterbeck & Glaser, *Proc. Natl. Acad. Sci. USA* 100 (2003) 5172–5177 — cross-correlated relaxation-optimised pulses.
- **Spinach**: [https://spindynamics.org/wiki/index.php?title=Main_Page](https://spindynamics.org/wiki/index.php?title=Main_Page) — MATLAB spin dynamics library whose `grape_xy` / `control` struct API this package mirrors.

## Local development commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Lint
ruff check .

# Typecheck
mypy optimalcontrol

# Test
python3 -m pytest
```

## Versioning policy

This package follows semantic versioning: `MAJOR.MINOR.PATCH`.

- `PATCH` releases contain backwards-compatible fixes only and must not introduce breaking API,
  file-format, or numerical-contract changes.
- `MINOR` releases may add features and deprecate existing APIs; deprecations must emit warnings
  before removal.
- `MAJOR` releases may remove deprecated APIs or introduce intentional breaking changes, with
  migration notes recorded in `CHANGELOG.md`.
