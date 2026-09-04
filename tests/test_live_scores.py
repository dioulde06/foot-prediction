"""The live-score patch, as the browser actually runs it.

The logic lives in the page (no server sits between ESPN and the visitor), so
the test extracts that block from the template and runs it under node rather
than restating it in Python, where it would drift.

What it guards: a match in progress shows a score but never settles a bet, a
committed result always wins over the live one, and a name ESPN spells its own
way still finds its fixture.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.app.site import TEMPLATE

START = "/* ---------------- live scores ---------------- */"
END = "/* ---------------- boot ---------------- */"

STUBS = """
const assert = require("assert");
let toasted = null, rendered = 0;
const document = {
  addEventListener() {},
  getElementById: () => ({ hidden: false, innerHTML: "" }),
  hidden: false,
};
const toast = (msg) => { toasted = msg; };
const rerender = () => { rendered++; };
const renderCarnet = () => {}, renderCert = () => {};
const kick = (m) => new Date(m.kickoff);
const stateOf = (m) => m.result ? "done" : "open";
const countdown = () => "";
let NOW = Date.now();
const DATA = __DATA__;
"""

CHECKS = """
const byName = (h, a) => DATA.upcoming.find((m) => m.home === h && m.away === a);

// A match in progress: score shown, minute shown, nothing settled.
let goals = patchLive([event("Manchester United", "Liverpool", 1, 0, "in")]);
let m = byName("Man United", "Liverpool");
assert.strictEqual(m.score, "1-0");
assert.strictEqual(m.result, null, "a match in progress must not settle a bet");
assert.ok(m.live, "a match in progress must carry its minute");
assert.strictEqual(goals, 0, "a first score seen is not a goal event");
assert.strictEqual(liveMiss, 0);

// The next goal is announced, and the page knows it has to redraw.
goals = patchLive([event("Manchester United", "Liverpool", 2, 0, "in")]);
assert.strictEqual(goals, 1, "a score that moves is a goal");
assert.strictEqual(byName("Man United", "Liverpool").score, "2-0");

// Full time: the result lands, flagged provisional until the parquet confirms.
patchLive([event("Manchester United", "Liverpool", 2, 1, "post")]);
m = byName("Man United", "Liverpool");
assert.strictEqual(m.result, "H");
assert.strictEqual(m.provisional, true);
assert.strictEqual(m.live, null, "a finished match is no longer live");

// A committed result is never overwritten; a disagreement is counted.
patchLive([event("Arsenal", "Chelsea", 0, 3, "post")]);
m = byName("Arsenal", "Chelsea");
assert.strictEqual(m.result, "H", "the committed result wins");
assert.strictEqual(m.score, "2-0", "the committed score wins");
assert.strictEqual(liveClash, 1, "the disagreement must be counted");

// A fixture we never published is not an anomaly; a club we cannot name is.
patchLive([event("Manchester United", "Chelsea", 1, 1, "post")]);
assert.strictEqual(liveMiss, 0, "an unpublished fixture is normal");
patchLive([event("Real Sociedad", "Deportivo", 1, 1, "post")]);
assert.strictEqual(liveMiss, 1, "an unknown club name is a missing mapping entry");

renderLive();
console.log("ok");
"""

EVENT = """
function event(home, away, hg, ag, state) {
  return { competitions: [{
    status: { displayClock: "60'", type: { state, completed: state === "post", shortDetail: "60'" } },
    competitors: [
      { homeAway: "home", score: String(hg), team: { displayName: home } },
      { homeAway: "away", score: String(ag), team: { displayName: away } },
    ],
  }] };
}
"""


def _data() -> dict[str, object]:
    from src.data.team_mapping import ESPN_LEAGUES, ESPN_SCOREBOARD, ESPN_TO_CANONICAL

    return {
        "meta": {
            "today": "2026-09-04",
            "live": {
                "base": ESPN_SCOREBOARD,
                "leagues": ESPN_LEAGUES,
                "names": ESPN_TO_CANONICAL,
                "windowDays": 10,
            },
        },
        "upcoming": [
            {
                "home": "Man United",
                "away": "Liverpool",
                "kickoff": "2026-09-04T14:00:00Z",
                "result": None,
                "score": None,
            },
            {
                "home": "Arsenal",
                "away": "Chelsea",
                "kickoff": "2026-09-03T14:00:00Z",
                "result": "H",
                "score": "2-0",
            },
        ],
    }


@pytest.mark.skipif(
    shutil.which("node") is None, reason="node is needed to run the page's own code"
)
def test_the_live_patch_shows_scores_without_settling_early(tmp_path: Path) -> None:
    page = TEMPLATE.read_text()
    block = page[page.index(START) : page.index(END)]
    script = tmp_path / "live.js"
    script.write_text(
        STUBS.replace("__DATA__", json.dumps(_data())) + EVENT + block + CHECKS
    )
    done = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout
