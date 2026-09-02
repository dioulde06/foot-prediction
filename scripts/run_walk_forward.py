"""Walk-forward validation and the training-history sweep.

Run: uv run python -m scripts.run_walk_forward   (or: make eval)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.walk_forward import (  # noqa: E402
    prepare,
    saturation,
    season_order,
    walk_forward,
)

LOG = logging.getLogger(__name__)

FIGDIR = Path("reports/figures")
REPORT = Path("reports/walk_forward.md")

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 120,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "grid.color": "#e8e7e3",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def plot_stability(table: pl.DataFrame) -> Path:
    seasons = table["test_season"].to_list()
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    # Model labels above its curve, market labels below its own, so neither
    # lands on the other curve or on the legend.
    for column, colour, label, dy in (
        ("market_log_loss", ORANGE, "market", -17),
        ("log_loss", BLUE, "modèle", 11),
    ):
        values = table[column].to_list()
        ax.plot(
            seasons,
            values,
            "o-",
            color=colour,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=label,
        )
        for season, value in zip(seasons, values, strict=True):
            ax.annotate(
                f"{value:.4f}",
                (season, value),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=INK,
            )
    gaps = table["gap_to_market"]
    ax.set_ylabel("log-loss  —  plus bas est meilleur")
    ax.margins(y=0.20)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.text(
        0.99,
        0.02,
        f"écart au marché : {float(gaps.min()):+.4f} à {float(gaps.max()):+.4f}",  # type: ignore[arg-type]
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color=INK_2,
    )
    ax.set_title(
        "Stabilité saison par saison, pas une moyenne",
        loc="left",
        fontweight="bold",
        pad=10,
    )
    plt.tight_layout()
    path = FIGDIR / "14_walk_forward_stabilite.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_saturation(table: pl.DataFrame, test_season: str) -> Path:
    counts = table["n_train_seasons"].to_list()
    values = table["log_loss"].to_list()
    market = table["market_log_loss"][0]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(market, color=ORANGE, linewidth=1.6, linestyle=(0, (4, 3)))
    ax.annotate(
        f"market  {market:.4f}",
        (counts[-1], market),
        xytext=(0, 7),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=ORANGE,
    )
    ax.plot(
        counts,
        values,
        "o-",
        color=BLUE,
        markersize=9,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    for count, value in zip(counts, values, strict=True):
        ax.annotate(
            f"{value:.4f}",
            (count, value),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xticks(counts)
    ax.set_xlabel("saisons d'entraînement")
    ax.set_ylabel("log-loss sur " + test_season)
    spread = max(values) - min(values)
    ax.set_ylim(market - 0.004, max(values) + 0.006)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.text(
        0.99,
        0.55,
        f"amplitude sur l'axe : {spread:.4f}\n"
        f"écart au marché : {min(values) - market:.4f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color=INK_2,
    )
    ax.set_title(
        "Où le gain d'historique sature", loc="left", fontweight="bold", pad=14
    )
    plt.tight_layout()
    path = FIGDIR / "15_saturation_historique.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown(stability: pl.DataFrame, sweep: pl.DataFrame, test_season: str) -> str:
    def table(frame: pl.DataFrame) -> str:
        header = "| " + " | ".join(frame.columns) + " |"
        rule = "|" + "|".join("---" for _ in frame.columns) + "|"
        rows = [
            "| " + " | ".join(str(v) for v in row) + " |" for row in frame.iter_rows()
        ]
        return "\n".join([header, rule, *rows])

    spread = float(stability["log_loss"].max()) - float(  # type: ignore[arg-type]
        stability["log_loss"].min()  # type: ignore[arg-type]
    )
    gap = stability["gap_to_market"]
    return (
        "\n".join(
            [
                "# Validation walk-forward",
                "",
                "Pour chaque saison de test : entraînement sur tout ce qui précède la",
                "saison de calibration, calibration sur la saison immédiatement",
                "antérieure, test sur la saison elle-même.",
                "",
                "## Stabilité",
                "",
                table(stability),
                "",
                f"Amplitude du log-loss sur les saisons testées : **{spread:.4f}**.",
                "Écart au marché : de "
                f"{float(gap.min()):+.4f} à {float(gap.max()):+.4f}, "  # type: ignore[arg-type]
                f"médiane {float(gap.median()):+.4f}.",  # type: ignore[arg-type]
                "",
                "Une moyenne n'est pas donnée volontairement : c'est la dispersion qui",
                "dit si le modèle est réel, et une moyenne la masque.",
                "",
                f"## Saturation de l'historique, testée sur {test_season}",
                "",
                table(sweep),
                "",
                "Prompt 5.1 demandait 2, 4 et 6 saisons. Avec six saisons de",
                "données dont une pour la calibration et une pour le test,",
                "quatre est le maximum : le balayage va donc de 1 à 4.",
                "",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-saturation", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    features, odds = prepare()
    seasons = season_order(features)

    stability = walk_forward(features, odds)
    sweep = (
        pl.DataFrame()
        if args.skip_saturation
        else saturation(features, odds, seasons[-1])
    )

    with pl.Config(tbl_hide_dataframe_shape=True, tbl_width_chars=200, tbl_cols=20):
        print("\n=== STABILITE ===")
        print(stability)
        if not sweep.is_empty():
            print(f"\n=== SATURATION (test {seasons[-1]}) ===")
            print(sweep)

    print("\nfigure :", plot_stability(stability))
    if not sweep.is_empty():
        print("figure :", plot_saturation(sweep, seasons[-1]))
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(markdown(stability, sweep, seasons[-1]))
        print("rapport :", REPORT)


if __name__ == "__main__":
    main()
