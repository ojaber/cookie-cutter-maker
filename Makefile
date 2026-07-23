PY ?= python3
VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest
PROJECT_ROOT := $(abspath .)
PIPELINE_OUTPUT_DIR ?= output
RELOAD ?=

.PHONY: venv install test run docker-build docker-up docker-down

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

test: install
	PYTHONPATH=$(PROJECT_ROOT) $(PYTEST) tests

run: install
	PIPELINE_OUTPUT_DIR=$(PIPELINE_OUTPUT_DIR) $(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	# Pre-create the bind-mounted dir so it's owned by you, not root — the
	# container runs as uid 1000 and must be able to write job files into it.
	mkdir -p output
	docker compose up --build

docker-down:
	docker compose down
