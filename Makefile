# ==============================================================
#						  CALL ME MAYBE
# ==============================================================

NAME = call_me_maybe
PYTHON = python3
UV = $(shell command -v uv 2> /dev/null || echo $(HOME)/.local/bin/uv)

DEPS =	flake8 \
		huggingface_hub \
		mypy \
		pudb \
		pydantic \
		torch \
		transformers

all: run

install:
	@if [ ! -e $(UV) ]; then \
		echo "installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; \
	fi
	@$(UV) sync >/dev/null 2>&1
	@$(UV) add $(DEPS)

run:
	@$(UV) run python -m src

debug:
	@$(UV) run python -m pudb -m src

lint:
	@echo "Running flake8..."
	@$(UV) run flake8 .
	@echo "Running mypy..."
	@$(UV) run mypy .


lint-strict:
	@echo "Running flake8..."
	@$(UV) run flake8 .
	@echo "Running mypy --strict..."
	@$(UV) run mypy . --strict

clean:
	@echo "Cleaning cache files..."
	@$(RM) -r .mypy_cache .pytest_cache .uv_cache __pycache__

fclean: clean
	@echo "Removing virtual environment..."
	@$(RM) -r .venv uv.lock

re: fclean all

.PHONY: install run debug clean fclean re all
.DEFAULT_GOAL = all
