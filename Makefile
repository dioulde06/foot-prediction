.PHONY: install lint test fetch baselines train eval

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

# Comparative table of the three baselines on the test season.
baselines:
	uv run python -m scripts.run_baselines $(if $(BOOK),--book $(BOOK),)

train:
	@echo "not implemented yet -- phase 4 (prompt 4.1)"; exit 1

eval:
	@echo "not implemented yet -- phase 5 (prompt 5.1)"; exit 1
