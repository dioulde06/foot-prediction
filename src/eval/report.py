"""Markdown report on the test season.

Everything here is computed on the test split, never on validation: the
calibration bins in particular must not be read on the rows that were used to
fit the calibrator, or the report describes the fit rather than the model.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.eval.metrics import (
    Probs,
    accuracy,
    brier_score_multiclass,
    calibration_bins,
    class_bias,
    expected_calibration_error,
    log_loss,
)

REPORTS_DIR = Path("reports")

# CLAUDE.md rule 2: past these, stop and look for the leak.
ACCURACY_ALARM = 0.60
LOG_LOSS_ALARM = 0.85


def _table(frame: pl.DataFrame, floats: int = 4) -> str:
    rounded = frame.with_columns(
        [pl.col(c).round(floats) for c, t in frame.schema.items() if t == pl.Float64]
    )
    header = "| " + " | ".join(rounded.columns) + " |"
    rule = "|" + "|".join("---" for _ in rounded.columns) + "|"
    rows = [
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        for row in rounded.iter_rows()
    ]
    return "\n".join([header, rule, *rows])


def scoreboard(named: dict[str, Probs], outcomes: Sequence[str]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "modele": name,
                "log_loss": log_loss(probs, outcomes),
                "brier": brier_score_multiclass(probs, outcomes),
                "ece": expected_calibration_error(probs, outcomes),
                "accuracy_info": accuracy(probs, outcomes),
            }
            for name, probs in named.items()
        ]
    )


def alarms(probs: Probs, outcomes: Sequence[str]) -> list[str]:
    """CLAUDE.md rule 2. A result past these bounds is a leak, not a success."""
    raised = []
    if accuracy(probs, outcomes) > ACCURACY_ALARM:
        raised.append(
            f"accuracy {accuracy(probs, outcomes):.4f} > {ACCURACY_ALARM}: "
            "stop and look for the temporal leak"
        )
    if log_loss(probs, outcomes) < LOG_LOSS_ALARM:
        raised.append(
            f"log-loss {log_loss(probs, outcomes):.4f} < {LOG_LOSS_ALARM}: "
            "stop and look for the temporal leak"
        )
    return raised


def markdown_report(
    named: dict[str, Probs],
    outcomes: Sequence[str],
    calibrated: Probs,
    metadata: dict[str, Any],
    importances: pl.DataFrame,
    entropy_floor: float,
) -> str:
    scores = scoreboard(named, outcomes)
    raised = alarms(calibrated, outcomes)

    lines = [
        "# Rapport modèle — saison de test " + metadata["test_season"],
        "",
        f"Généré le {dt.date.today().isoformat()}.",
        "",
        "## Ce sur quoi le modèle a été entraîné",
        "",
        f"- Saisons d'entraînement : {', '.join(metadata['train_seasons'])} "
        f"({metadata['n_train']} matchs, {metadata['train_dates'][0]} → "
        f"{metadata['train_dates'][1]})",
        f"- Saison de validation : {metadata['valid_season']} "
        f"({metadata['n_valid']} matchs) — sert à l'early stopping **et** à la "
        "calibration, jamais à mesurer",
        f"- Saison de test : {metadata['test_season']} ({len(outcomes)} matchs)",
        f"- Empreinte du dataset : `{metadata['dataset_hash']}`",
        f"- Arrêt à l'itération {metadata['best_iteration']}, log-loss "
        f"validation {metadata['valid_log_loss']:.4f}",
        f"- Features ({len(metadata['features'])}) : "
        + ", ".join(f"`{f}`" for f in metadata["features"]),
        "",
        "## Résultats sur le test",
        "",
        _table(scores),
        "",
        f"Repères : ne rien savoir vaut `ln(3) = {float(np.log(3)):.4f}`. "
        "L'entropie moyenne des probabilités du marché vaut "
        f"{entropy_floor:.4f}, "
        "ce qui estime le plancher sous lequel un log-loss n'est plus honnête.",
        "",
        "L'accuracy est affichée pour information et ne décide de rien.",
        "",
        "## Garde-fou",
        "",
    ]
    if raised:
        lines += ["**ALERTE.**", ""] + [f"- {r}" for r in raised]
    else:
        lines.append(
            "Aucune alerte : accuracy sous "
            f"{ACCURACY_ALARM} et log-loss au-dessus de {LOG_LOSS_ALARM}."
        )

    lines += [
        "",
        "## Calibration du modèle calibré, par tranche",
        "",
        "Calculée sur le **test**. La validation a servi à ajuster le "
        "calibrateur, la lire ici décrirait l'ajustement et non le modèle.",
        "",
        _table(calibration_bins(calibrated, outcomes)),
        "",
        f"ECE = {expected_calibration_error(calibrated, outcomes):.4f}, "
        "soit l'erreur moyenne en points de pourcentage.",
        "",
        "## Biais signé par classe",
        "",
        "Le biais global est identiquement nul en multiclasse normalisé : les "
        "lignes de probabilités et les issues réalisées somment toutes deux à 1. "
        "Seul le biais par classe informe.",
        "",
        _table(class_bias(calibrated, outcomes)),
        "",
        "## Importance des features",
        "",
        _table(importances, floats=1),
        "",
    ]
    return "\n".join(lines) + "\n"


def write_report(text: str, name: str = "model_report.md") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / name
    path.write_text(text)
    return path
