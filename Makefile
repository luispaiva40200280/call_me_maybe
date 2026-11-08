FUNCTIONS ?= ./data/input/functions_definition.json
INPUT ?= ./data/input/function_calling_tests.json
OUTPUT ?= ./data/output/function_calls.json

func_moul ?= ../data/input/functions_definition.json
imp_moul ?= ../data/input/function_calling_tests.json
outp_moul ?= ../data/output/result.json

all: install run

install:
	uv sync

run:
	uv run python -m src --functions_definition $(FUNCTIONS) --input $(INPUT) --output $(OUTPUT)

debug:
	uv run python -m src --engine generator

debug-moulinete:
	uv run python -m src  --functions_definition $(func_moul) --input $(imp_moul) --output $(outp_moul) --engine generator


clean:
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "Cleaning all project cache files..."
# 1. Finds and deletes all __pycache__ folders anywhere
	@find . -type d -name "__pycache__" -exec rm -rf {} +
# 2. Deletes lingering compiled python files anywhere
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
# 3. Recursively removes Mypy and Pytest cache folders anywhere
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +

lint:
	flake8 src
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs srcmake

.PHONY = install run clean debug lint


#uv venv
#uv pip install  pydantic numpy flake8 
#uv pip install -e llm_sdk