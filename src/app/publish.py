"""Publish predictions before a matchday, and reconcile them afterwards.

Two properties make the whole exercise honest, and both are mechanical rather
than declarative:

1. **Append-only.** predictions/predictions.parquet is only ever appended to.
   A published row is never edited, never deleted. Reconciliation writes
   nothing back into it; it joins results on the fly.
2. **Timestamped by something we do not control.** Each row carries a
   published_at and the payload's own sha256, and the file is committed to git.
   A sceptic does not have to trust the timestamps in the file: git's history
   says when the prediction existed, and the hash says it has not changed.

Features for an upcoming fixture go through build_features exactly like a
training row. There is no second code path, so there is no train/serve skew.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.app.scorers import GoalsPrior, append_scorers, scorers_for_fixtures
from src.data.fetch import LEAGUES, _download, season_of
from src.data.players import fetch_players
from src.data.schedule import fetch_schedule, upcoming
from src.eval.metrics import CLASSES, Probs
from src.features.build import FEATURE_COLUMNS, build_features
from src.models.calibrate import TemperatureScaler
from src.models.train import MATCHES_PARQUET, MODEL_DIR

LOG = logging.getLogger(__name__)

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
PREDICTIONS_DIR = Path("predictions")
PREDICTIONS_PARQUET = PREDICTIONS_DIR / "predictions.parquet"

# Columns of a published row. Fixed, because appending to a file whose schema
# drifts is how an append-only history quietly becomes unreadable.
SCHEMA: dict[str, pl.DataType] = {
    "published_at": pl.Datetime("us"),
    "payload_sha256": pl.String(),
    "model_hash": pl.String(),
    "date": pl.Date(),
    "league": pl.String(),
    "home_team": pl.String(),
    "away_team": pl.String(),
    "p_home": pl.Float64(),
    "p_draw": pl.Float64(),
    "p_away": pl.Float64(),
    "temperature": pl.Float64(),
}

# How far ahead fixtures are predicted. The schedule gives the whole season;
# a month is what the page shows and what the model's inputs can still say
# something about.
HORIZON_DAYS = 35

# Books carried by the fixtures feed, as column prefixes there and here.
FEED_BOOKS: dict[str, str] = {
    "B365": "Bet365",
    "BFD": "Betfair",
    "BV": "BetVictor",
    "BW": "Bwin",
    "PP": "Paddy Power",
    "SKB": "Sky Bet",
}

# The odds of a fixture, captured once at publication time. A separate file:
# the prediction log's schema is published and never changes, and the odds
# are context, not a prediction. The market average is required; a book that
# has not priced a match yet reads as null.
ODDS_PARQUET = PREDICTIONS_DIR / "odds.parquet"
ODDS_SCHEMA: dict[str, pl.DataType] = {
    "captured_at": pl.Datetime("us"),
    "date": pl.Date(),
    "league": pl.String(),
    "home_team": pl.String(),
    "away_team": pl.String(),
    "kickoff_time": pl.String(),
    "odds_avg_h": pl.Float64(),
    "odds_avg_d": pl.Float64(),
    "odds_avg_a": pl.Float64(),
    **{f"odds_{code.lower()}_{o}": pl.Float64() for code in FEED_BOOKS for o in "hda"},
}
KEY = ["date", "home_team", "away_team"]


def fetch_fixtures() -> pl.DataFrame:
    """Upcoming fixtures for the big five, from the football-data feed.

    The feed carries current odds, not closing odds -- at publication time the
    closing line does not exist yet, which is exactly the point.
    """
    raw = _download(FIXTURES_URL)
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    frame = pl.read_csv(raw, infer_schema_length=0, encoding="utf8-lossy")

    needed = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "AvgH", "AvgD", "AvgA"]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"fixtures feed is missing {missing}")

    fixtures = (
        frame.filter(pl.col("Div").str.strip_chars().is_in(list(LEAGUES)))
        .select(
            pl.col("Div").str.strip_chars().alias("league_code"),
            pl.when(pl.col("Date").str.len_chars() == 10)
            .then(pl.col("Date").str.to_date("%d/%m/%Y"))
            .otherwise(pl.col("Date").str.to_date("%d/%m/%y"))
            .alias("date"),
            pl.col("HomeTeam").str.strip_chars().alias("home_team"),
            pl.col("AwayTeam").str.strip_chars().alias("away_team"),
            pl.col("Time").str.strip_chars().alias("kickoff_time"),
            # Empty cells are legitimate: a book may not have priced a match yet,
            # and a book absent from the file altogether reads as null.
            *[
                _odds_column(frame, f"{code}{o.upper()}", f"odds_{code.lower()}_{o}")
                for code in ["Avg", *FEED_BOOKS]
                for o in "hda"
            ],
        )
        .with_columns(pl.col("league_code").replace_strict(LEAGUES).alias("league"))
        .sort("date", "home_team")
    )
    LOG.info("%d matchs a venir dans les 5 championnats", fixtures.height)
    return fixtures


def _odds_column(frame: pl.DataFrame, source: str, target: str) -> pl.Expr:
    if source not in frame.columns:
        return pl.lit(None, dtype=pl.Float64).alias(target)
    return (
        pl.col(source)
        .str.strip_chars()
        .replace("", None)
        .cast(pl.Float64)
        .alias(target)
    )


def _conform(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """`frame` with exactly the schema's columns; missing ones are null."""
    return frame.with_columns(
        [
            pl.lit(None, dtype=t).alias(c)
            for c, t in schema.items()
            if c not in frame.columns
        ]
    ).select([pl.col(c).cast(t) for c, t in schema.items()])


def append_odds(fixtures: pl.DataFrame, captured_at: dt.datetime) -> pl.DataFrame:
    """Record the odds of every priced fixture in the feed, once. Never rewrites.

    The first capture wins: it is the price that existed when the prediction
    was published, which is the only one a sceptic can compare it to. A fixture
    the market has not priced yet is skipped, not recorded as empty, so it is
    captured on the day it gets a price.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _conform(
        fixtures.with_columns(
            pl.lit(captured_at).cast(pl.Datetime("us")).alias("captured_at")
        ),
        ODDS_SCHEMA,
    ).filter(pl.all_horizontal([pl.col(f"odds_avg_{o}").is_not_null() for o in "hda"]))
    if ODDS_PARQUET.exists():
        # Older files predate the per-book columns: widen them with nulls, and
        # write the widened file even when there is nothing new to add.
        stored = pl.read_parquet(ODDS_PARQUET)
        history = _conform(stored, ODDS_SCHEMA)
        if list(stored.columns) != list(ODDS_SCHEMA):
            history.write_parquet(ODDS_PARQUET)
        rows = rows.join(history.select(KEY), on=KEY, how="anti")
        if rows.is_empty():
            return history
        combined = pl.concat([history, rows])
    else:
        combined = rows
    if combined.is_empty():
        return combined
    combined.write_parquet(ODDS_PARQUET)
    LOG.info("%d nouvelles cotes capturees dans %s", rows.height, ODDS_PARQUET)
    return combined


def load_model(directory: Path = MODEL_DIR) -> tuple[lgb.Booster, float, str]:
    """Booster, calibration temperature, and a hash of the saved model file."""
    model_path = directory / "model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found; run `make train` first")
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["features"] != list(FEATURE_COLUMNS):
        raise ValueError(
            "the saved model was trained on different features:\n"
            f"  saved   {metadata['features']}\n  current {list(FEATURE_COLUMNS)}"
        )
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    return (
        lgb.Booster(model_file=str(model_path)),
        float(metadata["temperature"]),
        model_hash,
    )


def upcoming_features(
    fixtures: pl.DataFrame, played: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Features for unplayed fixtures, via the training code path.

    The fixtures are concatenated onto the played matches with null outcome
    columns. Rolling windows are shifted by one and Elo skips unplayed rows, so
    a fixture reads only matches that have actually happened.
    """
    if played is None:
        played = pl.read_parquet(MATCHES_PARQUET).sort("date")
    if fixtures.is_empty():
        return fixtures

    latest = played["date"].max()
    future = fixtures.filter(pl.col("date") > latest)
    if future.height < fixtures.height:
        LOG.warning(
            "%d matchs du flux sont deja dans le dataset, ignores",
            fixtures.height - future.height,
        )
    if future.is_empty():
        return future

    blanks = {
        column: pl.lit(None, dtype=played.schema[column]).alias(column)
        for column in played.columns
        if column not in ("date", "league", "home_team", "away_team")
    }
    # The season is derived from the fixture date, never copied from the last
    # played match: a fixture in August belongs to the new season, and getting
    # that wrong skips the between-season Elo regression entirely.
    padded = future.select(
        "date",
        "league",
        "home_team",
        "away_team",
        *[blanks[c] for c in played.columns if c in blanks],
    ).with_columns(
        pl.col("date").map_elements(season_of, return_dtype=pl.String).alias("season")
    )

    combined = pl.concat([played, padded.select(played.columns)]).sort("date")
    features = build_features(combined, combined["date"].max())  # type: ignore[arg-type]
    return features.filter(pl.col("date") > latest)


def publish(
    as_of: dt.datetime | None = None, directory: Path = MODEL_DIR
) -> pl.DataFrame:
    """Predict the upcoming fixtures and append them to the history."""
    booster, temperature, model_hash = load_model(directory)
    published_at = as_of or dt.datetime.now(dt.UTC).replace(tzinfo=None)
    feed = fetch_fixtures()
    if not feed.is_empty():
        append_odds(feed, published_at)
    fixtures = fixtures_ahead(feed, published_at.date())
    features = upcoming_features(fixtures)
    if features.is_empty():
        LOG.info("aucun match a venir, rien a publier")
        return pl.DataFrame(schema=SCHEMA)

    matrix = features.select(FEATURE_COLUMNS).to_numpy()
    raw = np.asarray(booster.predict(matrix), dtype=np.float64)
    scaler = TemperatureScaler()
    scaler._temperature = temperature  # noqa: SLF001 -- restored, not refitted
    probs: Probs = scaler.transform(raw)

    rows = features.select("date", "league", "home_team", "away_team").with_columns(
        pl.Series("p_home", probs[:, CLASSES.index("H")]),
        pl.Series("p_draw", probs[:, CLASSES.index("D")]),
        pl.Series("p_away", probs[:, CLASSES.index("A")]),
        pl.lit(temperature).alias("temperature"),
        pl.lit(published_at).alias("published_at").cast(pl.Datetime("us")),
        pl.lit(model_hash).alias("model_hash"),
    )
    payload = hashlib.sha256(
        rows.drop("published_at").write_json().encode()
    ).hexdigest()[:32]
    rows = rows.with_columns(pl.lit(payload).alias("payload_sha256")).select(
        list(SCHEMA)
    )
    before = (
        pl.read_parquet(PREDICTIONS_PARQUET) if PREDICTIONS_PARQUET.exists() else None
    )
    history = append(rows)
    appended = history.tail(
        history.height - (before.height if before is not None else 0)
    )
    if appended.height:
        # Scorers follow the prediction: frozen again whenever it moves.
        freeze_scorers(
            features.join(appended.select(KEY), on=KEY, how="semi"), published_at
        )
    return history


def fixtures_ahead(feed: pl.DataFrame, today: dt.date) -> pl.DataFrame:
    """Every unplayed fixture within the horizon: the season schedule, plus
    whatever the odds feed lists that the schedule does not know yet."""
    played = pl.read_parquet(MATCHES_PARQUET)
    known = frozenset(played["home_team"]) | frozenset(played["away_team"])
    if not feed.is_empty():
        known = known | frozenset(feed["home_team"]) | frozenset(feed["away_team"])
    schedule = fetch_schedule(int(season_of(today)[:4]), known)
    ahead = upcoming(schedule, today, HORIZON_DAYS)
    if feed.is_empty():
        return ahead
    extra = feed.select("date", "league", "home_team", "away_team").join(
        ahead.select(KEY), on=KEY, how="anti"
    )
    if extra.height:
        LOG.info("%d matchs du flux absents du calendrier, ajoutes", extra.height)
    return pl.concat([ahead.drop("kickoff_utc"), extra]).sort("date", "home_team")


def freeze_scorers(features: pl.DataFrame, published_at: dt.datetime) -> pl.DataFrame:
    """Probable scorers for the same fixtures, frozen next to the predictions.

    Player stats are refreshed at every publication (the current season moves
    every week) over the current season and the one before, so a summer
    transfer shows under his new club as soon as he has played for it.
    """
    played = pl.read_parquet(MATCHES_PARQUET)
    current = int(season_of(published_at.date())[:4])
    players = fetch_players([current - 1, current], force=True)
    prior = GoalsPrior.from_data(played, players)
    return append_scorers(scorers_for_fixtures(features, players, prior), published_at)


# Probabilities are compared at this precision to decide whether a prediction
# moved. Below it the change is noise from the same information.
MOVED = 1e-4


def latest(history: pl.DataFrame) -> pl.DataFrame:
    """One row per match: the most recently published."""
    return history.sort("published_at").group_by(KEY, maintain_order=True).last()


def append(rows: pl.DataFrame) -> pl.DataFrame:
    """Append to the history. Never rewrites an existing row.

    A match is published again only when its probabilities moved, that is when
    new results changed its inputs. The latest row before kick-off is the
    prediction that counts; the earlier ones show how it travelled.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if PREDICTIONS_PARQUET.exists():
        history = pl.read_parquet(PREDICTIONS_PARQUET)
        current = latest(history).select(
            *KEY,
            *[pl.col(f"p_{k}").alias(f"last_{k}") for k in ("home", "draw", "away")],
        )
        rows = rows.join(current, on=KEY, how="left").filter(
            pl.col("last_home").is_null()
            | pl.any_horizontal(
                [
                    (pl.col(f"p_{k}") - pl.col(f"last_{k}")).abs() > MOVED
                    for k in ("home", "draw", "away")
                ]
            )
        )
        skipped = current.height - rows.join(current, on=KEY, how="semi").height
        if skipped > 0:
            LOG.info("%d matchs deja publies et inchanges, pas republies", skipped)
        if rows.is_empty():
            return history
        combined = pl.concat([history, rows.select(list(SCHEMA))])
    else:
        combined = rows.select(list(SCHEMA))

    combined.write_parquet(PREDICTIONS_PARQUET)
    LOG.info(
        "%d nouvelles predictions, %d au total dans %s",
        rows.height,
        combined.height,
        PREDICTIONS_PARQUET,
    )
    return combined


def reconcile(played: pl.DataFrame | None = None) -> pl.DataFrame:
    """Join the published history with the results, without touching the file."""
    if not PREDICTIONS_PARQUET.exists():
        return pl.DataFrame()
    if played is None:
        played = pl.read_parquet(MATCHES_PARQUET)

    history = latest(pl.read_parquet(PREDICTIONS_PARQUET))
    return history.join(
        played.select(
            "date",
            "home_team",
            "away_team",
            "result",
            *[f"odds_close_avg_{o}" for o in "hda"],
        ),
        on=["date", "home_team", "away_team"],
        how="left",
    )
