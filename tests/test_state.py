"""The fixture state machine, run as the browser runs it.

stateOf decides whether a match reads as upcoming, being played or over. It
lives in the page's inline script, and it is what broke in production: a
fixture with no kick-off time fell back to `new Date(null)` -- the epoch --
and twenty matches, four of them days in the future, read "En cours" for good.
There was no upper bound either, so a match whose official result had not
landed yet stayed "En cours" indefinitely.

The block is extracted between its markers and evaluated under node, so this
checks the code the visitor actually runs rather than a copy of it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = Path("src/app/templates/index.html")
START, END = "/* STATE-LOGIC-START", "/* STATE-LOGIC-END */"

# 2026-09-18T10:00:00Z, the moment the bug was caught on the live page.
NOW = 1789725600000
HOUR = 3600_000


def state_logic() -> str:
    """The marked block of the template, without its surrounding page."""
    text = TEMPLATE.read_text()
    start, end = text.index(START), text.index(END)
    return text[start:end]


def states(matches: list[dict[str, object]]) -> list[str]:
    """stateOf applied to each match by node, at a frozen NOW."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - node ships with the CI image
        pytest.skip("node is needed to run the page's own state logic")
    script = (
        state_logic().replace("let NOW = Date.now();", f"let NOW = {NOW};", 1)
        + f"\nconsole.log(JSON.stringify({json.dumps(matches)}.map(stateOf)));"
    )
    done = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return list(json.loads(done.stdout))


def test_the_marked_block_is_self_contained() -> None:
    block = state_logic()
    assert "stateOf" in block and "kickMs" in block
    assert "DATA" not in block, "the block must not reach into the page's payload"


def test_a_fixture_without_a_kickoff_time_is_not_live_since_1970() -> None:
    """The bug as it stood: no kick-off time, so `new Date(null)` put the match
    at the epoch and every one of them read as being played."""
    assert states(
        [
            {"date": "2026-09-13", "kickoff": None, "result": None},  # five days ago
            {"date": "2026-09-20", "kickoff": None, "result": None},  # in two days
            {"date": "2026-09-18", "kickoff": None, "result": None},  # later today
        ]
    ) == ["done", "open", "open"]


def test_a_match_is_over_once_it_has_been_played_result_or_not() -> None:
    """Official results land twice a day. Between the final whistle and the next
    publication a match has no result and is still over."""
    started = NOW - 4 * HOUR
    assert states(
        [
            {"date": "2026-09-18", "kickoff": iso(started), "result": None},
            {"date": "2026-09-18", "kickoff": iso(started), "result": "H"},
        ]
    ) == ["done", "done"]


def test_a_match_on_the_clock_still_reads_live() -> None:
    assert states(
        [
            {"date": "2026-09-18", "kickoff": iso(NOW - HOUR), "result": None},
            {"date": "2026-09-18", "kickoff": iso(NOW + HOUR), "result": None},
        ]
    ) == ["live", "open"]


def test_a_published_result_wins_over_the_clock() -> None:
    """A result is final whatever the clock says, and it is what settles a bet."""
    assert states(
        [{"date": "2026-09-30", "kickoff": iso(NOW + 99 * HOUR), "result": "D"}]
    ) == ["done"]


def iso(ms: int) -> str:
    """Milliseconds since the epoch as the ISO text the payload carries."""
    import datetime as dt

    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).strftime("%Y-%m-%dT%H:%M:00Z")
