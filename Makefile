
install:
	uv venv
	uv pip install pydantic numpy

run:
	uv run python -m src
debug:
	uv run python -m pdb -m src
clean:
	rm -rf __pycache__ .mypy_cache

lint:
	flake8 src
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs srcmake
