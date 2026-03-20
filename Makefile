# ==============================================================
#						  CALL ME MAYBE
# ==============================================================

NAME = call_me_maybe
PYTHON = python3
UV = $(shell command -v uv 2> /dev/null || echo $(HOME)/.local/bin/uv)
ENV = --env-file .env
SRC = src

DEPS =	accelerate \
		flake8 \
		huggingface_hub \
		mypy \
		pudb \
		pydantic \
		regex \
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
	@$(UV) run ${ENV} python -m $(SRC)

debug:
	@$(UV) run ${ENV} python -m pudb -m $(SRC)

lint:
	@echo "Running flake8..."
	@$(UV) run flake8 $(SRC)
	@echo "Running mypy..."
	@$(UV) run mypy $(SRC)


lint-strict:
	@echo "Running flake8..."
	@$(UV) run flake8 $(SRC)
	@echo "Running mypy --strict..."
	@$(UV) run mypy $(SRC) --strict

clean:
	@echo "Cleaning cache files..."
	@$(RM) -r .mypy_cache .pytest_cache .uv_cache __pycache__

fclean: clean
	@echo "Removing virtual environment..."
	@$(RM) -r .venv uv.lock

re: fclean all

.PHONY: all install run debug lint lint-strict clean fclean re
.DEFAULT_GOAL = all
