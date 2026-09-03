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

from src.data.fetch import LEAGUES, _download, season_of
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


def fetch_fixtures() -> pl.DataFrame:
    """Upcoming fixtures for the big five, from the football-data feed.

    The feed carries current odds, not closing odds -- at publication time the
    closing line does not exist yet, which is exactly the point.
    """
    raw = _download(FIXTURES_URL)
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    frame = pl.read_csv(raw, infer_schema_length=0, encoding="utf8-lossy")

    needed = ["Div", "Date", "HomeTeam", "AwayTeam"]
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
        )
        .with_columns(pl.col("league_code").replace_strict(LEAGUES).alias("league"))
        .sort("date", "home_team")
    )
    LOG.info("%d matchs a venir dans les 5 championnats", fixtures.height)
    return fixtures


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
    features = upcoming_features(fetch_fixtures())
    if features.is_empty():
        LOG.info("aucun match a venir, rien a publier")
        return pl.DataFrame(schema=SCHEMA)

    matrix = features.select(FEATURE_COLUMNS).to_numpy()
    raw = np.asarray(booster.predict(matrix), dtype=np.float64)
    scaler = TemperatureScaler()
    scaler._temperature = temperature  # noqa: SLF001 -- restored, not refitted
    probs: Probs = scaler.transform(raw)

    published_at = as_of or dt.datetime.now(dt.UTC).replace(tzinfo=None)
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

    return append(rows)


def append(rows: pl.DataFrame) -> pl.DataFrame:
    """Append to the history. Never rewrites an existing row."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if PREDICTIONS_PARQUET.exists():
        history = pl.read_parquet(PREDICTIONS_PARQUET)
        already = history.join(
            rows.select("date", "home_team", "away_team"),
            on=["date", "home_team", "away_team"],
            how="semi",
        )
        if already.height:
            LOG.warning(
                "%d matchs deja publies, ils ne sont pas republies", already.height
            )
            rows = rows.join(
                history.select("date", "home_team", "away_team"),
                on=["date", "home_team", "away_team"],
                how="anti",
            )
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

    history = pl.read_parquet(PREDICTIONS_PARQUET)
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
