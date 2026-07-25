"""Keep requirements-render.txt in sync with requirements.txt.

The Render free-tier deploy installs requirements-render.txt, so a dependency
added to requirements.txt but not mirrored here would import fine locally and
in CI, then crash the deployed app with an ImportError. This test fails first
instead.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Deliberately absent from the slim deploy list:
#   rembg[cpu] — the ~170 MB U2Net photo-background-removal model; needs more
#                RAM than a 512 MB free instance has (REMBG_ENABLED=false there).
#   pytest     — test-only, never imported at runtime.
RENDER_EXCLUSIONS = {"rembg", "pytest"}


def _requirements(path: Path) -> set[str]:
    """Package names (lowercased, extras/version specifiers stripped)."""
    names = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = line.split("[", 1)[0]
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            name = name.split(sep, 1)[0]
        names.add(name.strip().lower())
    return names


def test_render_requirements_match_expected_subset():
    full = _requirements(REPO_ROOT / "requirements.txt")
    render = _requirements(REPO_ROOT / "requirements-render.txt")

    missing = (full - RENDER_EXCLUSIONS) - render
    assert not missing, (
        f"requirements.txt gained {sorted(missing)} but requirements-render.txt "
        "was not updated — the Render deploy would crash with ImportError. Add "
        "them there, or to RENDER_EXCLUSIONS if intentionally omitted."
    )

    extra = render - full
    assert not extra, (
        f"requirements-render.txt has {sorted(extra)} not present in "
        "requirements.txt — the two lists have drifted."
    )

    unexpected = render & RENDER_EXCLUSIONS
    assert not unexpected, (
        f"{sorted(unexpected)} is excluded from the slim deploy list but appears "
        "in requirements-render.txt."
    )


def test_openai_is_installed_on_render():
    """app/main.py imports OpenAIError unconditionally, so the deploy needs the
    package even when the prompt feature is unused."""
    assert "openai" in _requirements(REPO_ROOT / "requirements-render.txt")
