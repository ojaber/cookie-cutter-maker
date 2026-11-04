PY ?= python3
VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest
PROJECT_ROOT := $(abspath .)
RELOAD ?=

.PHONY: venv install test run docker-build docker-up docker-down

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

test: install
	PYTHONPATH=$(PROJECT_ROOT) $(PYTEST) tests

run: install
	$(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down
