.PHONY: install lint test clean run

install:
	pip install -r requirements.txt

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache

run:
	python -m src.run_pipeline
