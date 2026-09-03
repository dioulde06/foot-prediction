"""Generate the static site from the committed parquet files.

The page is a pure function of the tracked files: the prediction log, the
odds captured at publication, the frozen scorers, the season schedule, the
played matches, and the walk-forward out-of-sample predictions. Nothing is
computed at request time: `make site` writes one HTML file and GitHub Pages
serves it. Everything the page computes (combinations, ticket, bet log) runs
in the browser on that data.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from src.app.combos import OBJECTIVES, PROPOSALS_PARQUET, PROPOSALS_SCHEMA
from src.app.publish import (
    FEED_BOOKS,
    HORIZON_DAYS,
    KEY,
    ODDS_PARQUET,
    ODDS_SCHEMA,
    PREDICTIONS_PARQUET,
    latest,
)
from src.app.scorers import (
    MIN_MINUTES,
    PRIOR_MINUTES,
    SCORERS_PARQUET,
    SCORERS_SCHEMA,
    goal_markets,
)
from src.data.schedule import SCHEDULE_PARQUET
from src.eval.baselines import devig_power
from src.eval.metrics import (
    BIN_LABELS,
    CLASSES,
    DEFAULT_BINS,
    calibration_bins,
    expected_calibration_error,
    log_loss,
)
from src.models.train import MATCHES_PARQUET

LOG = logging.getLogger(__name__)

TEMPLATE = Path("src/app/templates/index.html")
BACKTEST_PARQUET = Path("reports/proposals_backtest.parquet")
# Notional stake per proposal in the track record; the page lets the visitor change it.
TRACK_STAKE = 10.0
RESULT_TO_PICK = {"H": "1", "D": "X", "A": "2"}
OOS_PARQUET = Path("reports/oos_predictions.parquet")
SITE_DIR = Path("site")
PLACEHOLDER = "__DATA__"
# Played matches stay on the page this long, so a bet log can settle itself.
KEEP_PLAYED_DAYS = 10
# The odds feed quotes UK time; used only when the schedule has no kick-off.
FEED_TZ = ZoneInfo("Europe/London")
BOOK_URLS = {
    "B365": "https://www.bet365.com",
    "BFD": "https://www.betfair.com",
    "BV": "https://www.betvictor.com",
    "BW": "https://www.bwin.com",
    "PP": "https://www.paddypower.com",
    "SKB": "https://skybet.com",
}
SCHEDULE_SCHEMA: dict[str, pl.DataType] = {
    "league": pl.String(),
    "season": pl.String(),
    "kickoff_utc": pl.Datetime("us"),
    "date": pl.Date(),
    "home_team": pl.String(),
    "away_team": pl.String(),
    "played": pl.Boolean(),
}


def _probs(frame: pl.DataFrame, prefix: str) -> np.ndarray:
    return frame.select([f"{prefix}{c.lower()}" for c in CLASSES]).to_numpy()


def _market(odds: list[float | None]) -> tuple[list[float] | None, float | None]:
    """Devigged market and overround, or None when a price is missing."""
    if any(o is None for o in odds):
        return None, None
    array = np.asarray([odds], dtype=np.float64)
    probs = devig_power(array)[0]
    return [round(float(p), 4) for p in probs], round(float((1 / array).sum() - 1), 4)


def _kickoff(row: dict[str, Any]) -> str | None:
    """UTC kick-off as ISO text: from the schedule, else from the feed's UK time."""
    if row.get("kickoff_utc") is not None:
        return str(row["kickoff_utc"].strftime("%Y-%m-%dT%H:%M:00Z"))
    if row.get("kickoff_time"):
        hour, minute = (int(x) for x in row["kickoff_time"].split(":"))
        local = dt.datetime.combine(row["date"], dt.time(hour, minute), tzinfo=FEED_TZ)
        return local.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:00Z")
    return None


def _form(played: pl.DataFrame, team: str, before: dt.date, n: int = 5) -> list[str]:
    """Last n results of `team` before `before`, oldest first, as V / N / D."""
    rows = (
        played.filter(
            ((pl.col("home_team") == team) | (pl.col("away_team") == team))
            & (pl.col("date") < before)
        )
        .sort("date")
        .tail(n)
    )
    out = []
    for r in rows.iter_rows(named=True):
        mine = r["home_goals"] if r["home_team"] == team else r["away_goals"]
        theirs = r["away_goals"] if r["home_team"] == team else r["home_goals"]
        out.append("V" if mine > theirs else "N" if mine == theirs else "D")
    return out


def _cards(
    match: dict[str, Any], scorers: pl.DataFrame, played: pl.DataFrame
) -> dict[str, Any] | None:
    rows = scorers.filter(
        (pl.col("date") == dt.date.fromisoformat(match["date"]))
        & (pl.col("home_team") == match["home"])
        & (pl.col("away_team") == match["away"])
    )
    if rows.is_empty():
        return None
    # The latest freeze of this match only.
    rows = rows.filter(pl.col("published_at") == rows["published_at"].max())
    sides: dict[str, Any] = {}
    lam: dict[str, float] = {}
    for side in ("home", "away"):
        part = rows.filter(pl.col("side") == side).sort("p_scores", descending=True)
        if part.is_empty():
            return None
        team = part.row(0, named=True)
        lam[side] = team["lambda_total"]
        sides[side] = {
            "team": team["team"],
            "elo": round(team["elo"]),
            "form": _form(played, team["team"], dt.date.fromisoformat(match["date"])),
            "points5": round(5 * team["form_points_5"], 1),
            "gf": round(team["goals_scored_5"], 2),
            "ga": round(team["goals_conceded_5"], 2),
            "xf": round(team["np_xg_created_5"], 2),
            "xa": round(team["np_xg_conceded_5"], 2),
            "scorers": [
                {
                    "player": r["player"],
                    "np_goals": r["np_goals"],
                    "np_xg": round(r["np_xg"], 1),
                    "minutes": r["minutes"],
                    "p_scores": round(r["p_scores"], 4),
                }
                for r in part.iter_rows(named=True)
            ],
        }
    markets = goal_markets(lam["home"], lam["away"])
    return {
        **sides,
        "lambdaHome": round(lam["home"], 2),
        "lambdaAway": round(lam["away"], 2),
        "btts": round(markets["btts"], 4),
        "over25": round(markets["over25"], 4),
    }


def _matches(
    joined: pl.DataFrame, scorers: pl.DataFrame, played: pl.DataFrame, today: dt.date
) -> list[dict[str, Any]]:
    rows = joined.filter(
        (pl.col("date") >= today - dt.timedelta(days=KEEP_PLAYED_DAYS))
        & (pl.col("date") <= today + dt.timedelta(days=HORIZON_DAYS))
    ).sort("date", "kickoff_utc", "home_team", nulls_last=True)
    out = []
    for i, r in enumerate(rows.iter_rows(named=True)):
        avg = [r["odds_avg_h"], r["odds_avg_d"], r["odds_avg_a"]]
        market, overround = _market(avg)
        books = None
        if market is not None:
            books = {}
            for code in FEED_BOOKS:
                prices = [r[f"odds_{code.lower()}_{o}"] for o in "hda"]
                if all(p is not None for p in prices):
                    books[code] = [float(p) for p in prices]
        match = {
            "id": f"m{i}",
            "date": r["date"].isoformat(),
            "kickoff": _kickoff(r),
            "league": r["league"],
            "home": r["home_team"],
            "away": r["away_team"],
            "model": [round(r[f"p_{k}"], 4) for k in ("home", "draw", "away")],
            "odds": None if market is None else [float(o) for o in avg],
            "market": market,
            "overround": overround,
            "books": books,
            "result": r["result"],
            "score": (
                f"{r['home_goals']}-{r['away_goals']}"
                if r["home_goals"] is not None
                else None
            ),
            "publishedAt": r["published_at"].strftime("%Y-%m-%d %H:%M"),
        }
        match["cards"] = _cards(match, scorers, played)
        out.append(match)
    return out


def _books(odds: pl.DataFrame) -> list[dict[str, Any]]:
    """The books, with the margin each one actually showed on the captured odds."""
    out: list[dict[str, Any]] = [
        {"key": "AVG", "name": "Moyenne du marché", "url": None, "margin": None}
    ]
    for code, name in FEED_BOOKS.items():
        cols = [f"odds_{code.lower()}_{o}" for o in "hda"]
        priced = odds.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in cols]))
        margin = None
        if priced.height:
            inverse = priced.select(
                [(1 / pl.col(c)).alias(c) for c in cols]
            ).sum_horizontal()
            margin = round(float(inverse.to_numpy().mean()) - 1, 4)
        out.append(
            {"key": code, "name": name, "url": BOOK_URLS[code], "margin": margin}
        )
    return out


def _standing(oos: pl.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Season-by-season gap to the market and the last season's calibration."""
    seasons = (
        oos.group_by("season").agg(pl.col("date").min().alias("start")).sort("start")
    )["season"].to_list()
    table = []
    for season in seasons:
        held = oos.filter(pl.col("season") == season)
        outcomes = held["result"].to_list()
        model_ll = log_loss(_probs(held, "model_"), outcomes)
        market_ll = log_loss(_probs(held, "market_"), outcomes)
        table.append(
            {
                "season": season,
                "n": held.height,
                "model": round(model_ll, 4),
                "market": round(market_ll, 4),
                "gap": round(model_ll - market_ll, 4),
            }
        )
    last = oos.filter(pl.col("season") == seasons[-1])
    outcomes = last["result"].to_list()
    probs = _probs(last, "model_")
    bins = [
        {
            "label": row["tranche"],
            "lo": DEFAULT_BINS[k],
            "hi": DEFAULT_BINS[k + 1],
            "n": row["n"],
            "predicted": row["predit_moyen"],
            "observed": row["observe"],
        }
        for k, row in enumerate(calibration_bins(probs, outcomes).iter_rows(named=True))
    ]
    assert [b["label"] for b in bins] == list(BIN_LABELS)
    standing = {
        "seasons": table,
        "bins_season": seasons[-1],
        "ece": round(expected_calibration_error(probs, outcomes), 4),
    }
    return standing, bins


def _track_objectives(rows: pl.DataFrame, played: pl.DataFrame) -> dict[str, Any]:
    """Proposals grouped by objective and week, each leg settled when it can be.

    A proposal is won when every leg hit, lost as soon as one leg missed, and
    open otherwise. The page recomputes euros from these at any stake.
    """
    if rows.is_empty():
        return {objective: {"weeks": []} for objective in OBJECTIVES}
    joined = rows.join(played.select(*KEY, "result"), on=KEY, how="left").sort(
        "week", "objective", "rank", "leg"
    )
    out: dict[str, Any] = {}
    for objective in OBJECTIVES:
        weeks = []
        part = joined.filter(pl.col("objective") == objective)
        for week in sorted(part["week"].unique().to_list()):
            bets = []
            wf = part.filter(pl.col("week") == week)
            for rank in sorted(wf["rank"].unique().to_list()):
                legs_frame = wf.filter(pl.col("rank") == rank)
                legs = []
                odds = pq = pm = 1.0
                for r in legs_frame.iter_rows(named=True):
                    hit = (
                        None
                        if r["result"] is None
                        else RESULT_TO_PICK[r["result"]] == r["pick"]
                    )
                    legs.append(
                        {
                            "home": r["home_team"],
                            "away": r["away_team"],
                            "date": r["date"].isoformat(),
                            "pick": r["pick"],
                            "odds": round(r["odds"], 2),
                            "hit": hit,
                        }
                    )
                    odds *= r["odds"]
                    pq *= r["p_market"]
                    pm *= r["p_model"]
                hits = [leg["hit"] for leg in legs]
                won = False if False in hits else (True if all(hits) else None)
                bets.append(
                    {
                        "odds": round(odds, 2),
                        "pq": round(pq, 4),
                        "pm": round(pm, 4),
                        "won": won,
                        "legs": legs,
                    }
                )
            weeks.append({"week": week.isoformat(), "bets": bets})
        out[objective] = {"weeks": weeks}
    return out


def _track(
    backtest: pl.DataFrame, live: pl.DataFrame, played: pl.DataFrame
) -> dict[str, Any]:
    season = backtest["season"][0] if backtest.height else None
    since = live["week"].min() if live.height else None
    return {
        "stake": TRACK_STAKE,
        "backtest": {
            "season": season,
            "objectives": _track_objectives(backtest, played),
        },
        "live": {
            "since": since.isoformat() if isinstance(since, dt.date) else None,
            "objectives": _track_objectives(live, played),
        },
    }


def build_data(
    predictions: pl.DataFrame,
    odds: pl.DataFrame,
    played: pl.DataFrame,
    oos: pl.DataFrame,
    today: dt.date,
    scorers: pl.DataFrame | None = None,
    schedule: pl.DataFrame | None = None,
    backtest: pl.DataFrame | None = None,
    proposals: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Everything the page needs, as plain JSON-ready values."""
    if predictions.is_empty():
        raise ValueError("no published predictions; run `make publish` first")
    if scorers is None:
        scorers = pl.DataFrame(schema=SCORERS_SCHEMA)
    if schedule is None:
        schedule = pl.DataFrame(schema=SCHEDULE_SCHEMA)
    if backtest is None:
        backtest = pl.DataFrame(schema={**PROPOSALS_SCHEMA, "season": pl.String()})
    if proposals is None:
        proposals = pl.DataFrame(schema=PROPOSALS_SCHEMA)
    current = latest(predictions)
    joined = (
        current.join(
            odds.drop("league", "captured_at").unique(subset=KEY, keep="first"),
            on=KEY,
            how="left",
        )
        .join(schedule.select(*KEY, "kickoff_utc"), on=KEY, how="left")
        .join(
            played.select(
                *KEY,
                "home_goals",
                "away_goals",
                "result",
                *[f"odds_close_avg_{o}" for o in "hda"],
            ),
            on=KEY,
            how="left",
        )
    )
    newest = predictions.sort("published_at").row(-1, named=True)
    standing, bins = _standing(oos)
    captured = odds["captured_at"].max() if odds.height else None
    return {
        "meta": {
            "publishedAt": newest["published_at"].strftime("%Y-%m-%d %H:%M UTC"),
            "today": today.isoformat(),
            "modelHash": newest["model_hash"],
            "temperature": round(newest["temperature"], 4),
            "nPublished": current.height,
            "horizonDays": HORIZON_DAYS,
            "oddsCapturedAt": (
                captured.strftime("%Y-%m-%d %H:%M UTC")
                if isinstance(captured, dt.datetime)
                else None
            ),
            "books": _books(odds),
            "scorers": {"minMinutes": MIN_MINUTES, "priorMatches": PRIOR_MINUTES // 90},
            "resultsNote": (
                "Les résultats arrivent deux fois par jour, une fois les matchs joués."
            ),
        },
        "upcoming": _matches(joined, scorers, played, today),
        "standing": standing,
        "bins": bins,
        "track": _track(backtest, proposals, played),
    }


def render(template: str, data: dict[str, Any]) -> str:
    if PLACEHOLDER not in template:
        raise ValueError(f"template has no {PLACEHOLDER} placeholder")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # A literal "</script>" inside the JSON would end the script block early.
    return template.replace(PLACEHOLDER, payload.replace("</", "<\\/"))


def _read_or_empty(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=schema)


def main() -> Path:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    data = build_data(
        pl.read_parquet(PREDICTIONS_PARQUET),
        _read_or_empty(ODDS_PARQUET, ODDS_SCHEMA),
        pl.read_parquet(MATCHES_PARQUET),
        pl.read_parquet(OOS_PARQUET),
        dt.datetime.now(dt.UTC).date(),
        _read_or_empty(SCORERS_PARQUET, SCORERS_SCHEMA),
        _read_or_empty(SCHEDULE_PARQUET, SCHEDULE_SCHEMA),
        _read_or_empty(BACKTEST_PARQUET, {**PROPOSALS_SCHEMA, "season": pl.String()}),
        _read_or_empty(PROPOSALS_PARQUET, PROPOSALS_SCHEMA),
    )
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out = SITE_DIR / "index.html"
    out.write_text(render(TEMPLATE.read_text(), data))
    LOG.info(
        "%s ecrit: %d matchs, modele %s",
        out,
        len(data["upcoming"]),
        data["meta"]["modelHash"],
    )
    return out


if __name__ == "__main__":
    main()
