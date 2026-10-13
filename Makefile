.PHONY: help install test lint format clean train evaluate

help:
	@echo "Available commands:"
	@echo "  make install    Install dependencies from requirements.txt"
	@echo "  make test       Run the unit test suite"
	@echo "  make train      Train InceptionV3 model"
	@echo "  make evaluate   Evaluate trained model on test set"
	@echo "  make clean      Remove python caches and temporary build artifacts"

install:
	pip install -r requirements.txt

test:
	python -m unittest discover tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
