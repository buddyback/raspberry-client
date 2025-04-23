.PHONY: install-dev lint format check help

help:
	@echo "Available commands:"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make lint             Run linting tools without fixing issues"
	@echo "  make format           Automatically fix formatting with black and isort"
	@echo "  make check            Run a comprehensive code quality check"

install-dev:
	pip install --upgrade pip
	pip install flake8 pylint black mypy isort

lint:
	flake8 main.py detector/ utils/ config/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 main.py detector/ utils/ config/ --count --max-line-length=127 --statistics
	pylint --disable=C0111,R0903,C0103 --ignored-modules=cv2,mediapipe --fail-under=7.0 main.py detector/ utils/ config/
	mypy --ignore-missing-imports main.py detector/ utils/ config/

format:
	isort main.py detector/ utils/ config/
	black --line-length 120 main.py detector/ utils/ config/

check: install-dev lint
	isort --check --diff main.py detector/ utils/ config/
	black --line-length 120 --check --diff main.py detector/ utils/ config/