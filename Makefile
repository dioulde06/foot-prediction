.PHONY: install lint test fetch fetch-current merge audit-teams baselines train eval information proposals-backtest publish site track

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest -q

# Odds + results from football-data.co.uk. FORCE=1 to refetch.
fetch:
	uv run python -m src.data.fetch $(if $(FORCE),--force,)

# Only the season being played, then the merged dataset. What the daily job runs.
fetch-current:
	uv run python -m src.data.fetch --current

# Join the sources into data/processed/matches.parquet.
merge:
	uv run python -m src.data.merge $(if $(FORCE),--force,)

# Team names of each source, live feed included, and the ones still unmapped.
audit-teams:
	uv run python -m scripts.audit_teams

# Comparative table of the three baselines on the test season.
baselines:
	uv run python -m scripts.run_baselines $(if $(BOOK),--book $(BOOK),)

train:
	uv run python -m scripts.train_and_report

eval:
	uv run python -m scripts.run_walk_forward

# Does the model know anything the market does not? Blend test, walk-forward.
information:
	uv run python -m scripts.run_information

# Predict the upcoming fixtures and append them to predictions/.
publish:
	uv run python -m scripts.publish

# The combinator replayed on the last complete season, for the Bilan.
proposals-backtest:
	uv run python -m scripts.run_proposals_backtest

# Static site from the committed parquet files, served by GitHub Pages.
site:
	uv run python -m src.app.site

# Calibration of the published predictions, once their matches are played.
track:
	uv run python -m scripts.track_calibration
