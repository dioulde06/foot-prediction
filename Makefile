.PHONY: install lint test fetch merge audit-teams baselines train eval

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

# Join the sources into data/processed/matches.parquet.
merge:
	uv run python -m src.data.merge $(if $(FORCE),--force,)

# Team names of each source, and the ones that still need a mapping entry.
audit-teams:
	uv run python -m scripts.audit_teams

# Comparative table of the three baselines on the test season.
baselines:
	uv run python -m scripts.run_baselines $(if $(BOOK),--book $(BOOK),)

train:
	@echo "not implemented yet -- phase 4 (prompt 4.1)"; exit 1

eval:
	@echo "not implemented yet -- phase 5 (prompt 5.1)"; exit 1
