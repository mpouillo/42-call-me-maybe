*This project has been created as part of the 42 curriculum by mpouillo.*

# 42-Call-Me-Maybe

### Description

TODO

### Instructions

uv is necessary to run this project. Install with:

```shell
$> python3 -m pip install uv
```

Install and run the project using the provided Makefile:

```shell
$> make install
# Creating virtual environment and installing dependencies...

$> make run
# ...
```

To check source code for programmatic and stylistic errors, run:

```shell
$> make lint
# Running mypy...
# ...
$> make lint-strict
# Running mypy --strict...
# ...
```

To remove any temporary files (`__pycache__` or `.mypy_cache`), run:

```shell
$> make clean
# Cleaning cache files...
```

To clean up environment files, run:

```shell
$> make fclean
# Removing .venv directory...
```

### Resources

- [A Guide to Structured Generation Using Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Building a simple State Machine in Python.](https://dev.to/karn/building-a-simple-state-machine-in-python)
- [Introducing JSON](https://www.json.org/json-en.html)
- [Implement a Trie in Python](https://wangyy395.medium.com/implement-a-trie-in-python-e8dd5c5fde3a)
- [Regex101](https://regex101.com/)
- AI was used to
    - help understand basic LLM functionalities
    - explain how to implement constrained decoding
    - teach how to apply pydantic models to the project
