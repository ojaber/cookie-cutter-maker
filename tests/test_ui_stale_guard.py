"""Every setting that goes into an STL must invalidate the last one built.

The generated cutter is built from the settings that were on screen when
"Generate my cutter" was pressed. The UI keeps offering that file — and its
download links — until it is rebuilt, so a control that feeds `stlParams()`
without also being watched by `markStlDirty()` silently hands the user a file
that disagrees with the sliders in front of them (ask for 250 mm, download the
95 mm cutter). Adding a slider is exactly when that is easy to forget, so pin
the two lists together here rather than waiting for someone to print the wrong
size.
"""
from __future__ import annotations

import re
from pathlib import Path

INDEX_HTML = Path(__file__).resolve().parent.parent / "app" / "static" / "index.html"


def _source() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


def _watched_ids(src: str) -> set[str]:
    """The element ids `markStlDirty()` is wired up to."""
    match = re.search(r"const STL_SETTING_IDS = \[(.*?)\];", src, re.S)
    assert match, "STL_SETTING_IDS array not found in index.html"
    return set(re.findall(r"'([^']+)'", match.group(1)))


def _stl_param_ids(src: str) -> set[str]:
    """The element ids `stlParams()` reads when building the request."""
    match = re.search(r"\nfunction stlParams\(fd\) \{(.*?)\n\}", src, re.S)
    assert match, "stlParams() not found in index.html"
    body = match.group(1)
    ids = set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", body))
    # safeName() reads the file-name box, which decides what the download is
    # called — a rename leaves the held blob carrying the old name.
    if "safeName()" in body:
        ids.add("name")
    return ids


def test_every_stl_setting_invalidates_the_previous_cutter():
    src = _source()
    unwatched = _stl_param_ids(src) - _watched_ids(src)
    assert not unwatched, (
        f"stlParams() sends {sorted(unwatched)} but STL_SETTING_IDS does not "
        "watch them, so changing one leaves the previous cutter on offer as if "
        "it matched. Add them to STL_SETTING_IDS."
    )


def test_watched_settings_all_exist():
    src = _source()
    missing = {i for i in _watched_ids(src) if f'id="{i}"' not in src}
    assert not missing, (
        f"STL_SETTING_IDS names {sorted(missing)}, which no element in "
        "index.html has — the listener would throw on page load."
    )
