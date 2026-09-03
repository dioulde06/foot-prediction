"""Generate the static site from the committed parquet files.

The page is a pure function of four inputs, all tracked in git or rebuilt
from tracked sources: the prediction log, the odds captured at publication,
the played matches, and the walk-forward out-of-sample predictions. Nothing
is computed at request time: `make site` writes one HTML file and GitHub
Pages serves it. The page changes when we publish, never in between.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.app.publish import ODDS_PARQUET, ODDS_SCHEMA, PREDICTIONS_PARQUET
from src.app.scorers import (
    MIN_MINUTES,
    PRIOR_MINUTES,
    SCORERS_PARQUET,
    SCORERS_SCHEMA,
    goal_markets,
)
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
OOS_PARQUET = Path("reports/oos_predictions.parquet")
SITE_DIR = Path("site")
PLACEHOLDER = "__DATA__"
KEY = ["date", "home_team", "away_team"]


def _probs(frame: pl.DataFrame, prefix: str) -> np.ndarray:
    return frame.select([f"{prefix}{c.lower()}" for c in CLASSES]).to_numpy()


def _market(odds: list[float | None]) -> tuple[list[float] | None, float | None]:
    """Devigged market and overround, or None when a price is missing."""
    if any(o is None for o in odds):
        return None, None
    array = np.asarray([odds], dtype=np.float64)
    probs = devig_power(array)[0]
    return [round(float(p), 4) for p in probs], round(float((1 / array).sum() - 1), 4)


def _upcoming(joined: pl.DataFrame, today: dt.date) -> list[dict[str, Any]]:
    rows = joined.filter(pl.col("result").is_null() & (pl.col("date") >= today)).sort(
        "date", "kickoff_time", "home_team"
    )
    out = []
    for i, r in enumerate(rows.iter_rows(named=True)):
        odds = [r["odds_avg_h"], r["odds_avg_d"], r["odds_avg_a"]]
        market, overround = _market(odds)
        out.append(
            {
                "id": f"m{i}",
                "date": r["date"].isoformat(),
                "time": r["kickoff_time"] or "",
                "league": r["league"],
                "home": r["home_team"],
                "away": r["away_team"],
                "model": [round(r[f"p_{k}"], 4) for k in ("home", "draw", "away")],
                "odds": None if market is None else [float(o) for o in odds],
                "market": market,
                "overround": overround,
                "publishedAt": r["published_at"].strftime("%Y-%m-%d %H:%M"),
            }
        )
    return out


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


def _retro(joined: pl.DataFrame) -> list[dict[str, Any]]:
    rows = joined.filter(pl.col("result").is_not_null()).sort("date", descending=True)
    out = []
    for r in rows.iter_rows(named=True):
        closing = [r["odds_close_avg_h"], r["odds_close_avg_d"], r["odds_close_avg_a"]]
        market, _ = _market(closing)
        out.append(
            {
                "date": r["date"].isoformat(),
                "league": r["league"],
                "home": r["home_team"],
                "away": r["away_team"],
                "score": f"{r['home_goals']}-{r['away_goals']}",
                "result": r["result"],
                "model": [round(r[f"p_{k}"], 4) for k in ("home", "draw", "away")],
                "market": market,
            }
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


def build_data(
    predictions: pl.DataFrame,
    odds: pl.DataFrame,
    played: pl.DataFrame,
    oos: pl.DataFrame,
    today: dt.date,
    scorers: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Everything the page needs, as plain JSON-ready values."""
    if scorers is None:
        scorers = pl.DataFrame(schema=SCORERS_SCHEMA)
    if predictions.is_empty():
        raise ValueError("no published predictions; run `make publish` first")
    joined = predictions.join(
        odds.select(*KEY, "kickoff_time", "odds_avg_h", "odds_avg_d", "odds_avg_a"),
        on=KEY,
        how="left",
    ).join(
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
    latest = predictions.sort("published_at").row(-1, named=True)
    standing, bins = _standing(oos)
    upcoming = _upcoming(joined, today)
    for match in upcoming:
        match["cards"] = _cards(match, scorers, played)
    return {
        "meta": {
            "publishedAt": latest["published_at"].strftime("%Y-%m-%d %H:%M UTC"),
            "generatedAt": today.isoformat(),
            "modelHash": latest["model_hash"],
            "temperature": round(latest["temperature"], 4),
            "nPublished": predictions.height,
            "scorers": {"minMinutes": MIN_MINUTES, "priorMatches": PRIOR_MINUTES // 90},
        },
        "upcoming": upcoming,
        "retro": _retro(joined),
        "standing": standing,
        "bins": bins,
    }


def render(template: str, data: dict[str, Any]) -> str:
    if PLACEHOLDER not in template:
        raise ValueError(f"template has no {PLACEHOLDER} placeholder")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # A literal "</script>" inside the JSON would end the script block early.
    return template.replace(PLACEHOLDER, payload.replace("</", "<\\/"))


def main() -> Path:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    odds = (
        pl.read_parquet(ODDS_PARQUET)
        if ODDS_PARQUET.exists()
        else pl.DataFrame(schema=ODDS_SCHEMA)
    )
    scorers = (
        pl.read_parquet(SCORERS_PARQUET)
        if SCORERS_PARQUET.exists()
        else pl.DataFrame(schema=SCORERS_SCHEMA)
    )
    data = build_data(
        pl.read_parquet(PREDICTIONS_PARQUET),
        odds,
        pl.read_parquet(MATCHES_PARQUET),
        pl.read_parquet(OOS_PARQUET),
        dt.datetime.now(dt.UTC).date(),
        scorers,
    )
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out = SITE_DIR / "index.html"
    out.write_text(render(TEMPLATE.read_text(), data))
    LOG.info(
        "%s ecrit: %d matchs a venir, %d joues, modele %s",
        out,
        len(data["upcoming"]),
        len(data["retro"]),
        data["meta"]["modelHash"],
    )
    return out


if __name__ == "__main__":
    main()
