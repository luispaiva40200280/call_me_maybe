FUNCTIONS ?= ./data/input/functions_definition.json
INPUT ?= ./data/input/function_calling_tests.json
OUTPUT ?= ./data/output/function_calls.json

all: install run

install:
	uv venv
	uv pip install  pydantic numpy flake8 
	uv pip install -e llm_sdk
run:
	uv run python -m src --functions_definition $(FUNCTIONS) --input $(INPUT) --output $(OUTPUT)
clean:
	rm -rf __pycache__ .mypy_cache

lint:
	flake8 src
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs srcmake

.PHONY = install run clean debug lint