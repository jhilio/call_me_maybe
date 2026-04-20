# ######

# Vars
PYTHON = python3
FLAKE8 = $(PYTHON) -m flake8
MYPY   = $(PYTHON) -m mypy

# install dependancies

install:
	uv sync

run:
	uv run src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
	rm -rf .mypy_cache

debug:
	$(PYTHON) -m pdb src

lint:
	$(FLAKE8) src
	$(MYPY) src \
	--warn-return-any --warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs


.PHONY: install run clean debug lint
