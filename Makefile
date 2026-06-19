.PHONY: install lint test clean run

install:
	pip install -r requirements.txt

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path().rglob('__pycache__')]"
	python -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path().rglob('*.pyc')]"
	python -c "import pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache')]]"

run:
	python -m src.run_pipeline
