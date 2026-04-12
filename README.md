*This project has been created as part of the 42 curriculum by mpouillo.*

# 42-Call-Me-Maybe

- [Description](#description)
    - [Overview](#overview)
    - [Input files](#input-files)
    - [Output files](#input-files)
- [Instructions](#instructions)
- [Notes](#notes)
    - [Algorithm Explanation](#algorithm-explanation)
    - [Design decisions](#design-decisions)
    - [Performance analysis](#performance-analysis)
    - [Challenges faced](#challenges-faced)
    - [Testing strategy](#testing-strategy)
    - [Example usage](#example-usage)
- [Resources](#resources)

## Description

### Overview

Call-Me-Maybe is a project written in Python 3.10+. It is an introduction to constrained decoding in LLMs.

The goal is to parse prompts, formatted as JSON, and get a small LLM model (0.6B parameters) to provide correct output, structured as valid JSON. The output must be restricted to a list of valid functions provided in another JSON file.

Due to small language models like the one used in this project (Qwen/Qwen3-0.6B) being notoriously unreliable at generating structured output, the challenge is to achieve near 100% reliabilty despite this by implementing constrained decoding: a technique that guides the model's output token by token to guarantee valid structure without relying on prompting alone.

### Input files

`data/input/function_calling_tests.json` must contain a JSON array of natural language prompts that the system will process.

Example: `function_calling_tests.json`
```json
[
    {
        "prompt": "What is the sum of 2 and 3?"
    },
    {
        "prompt": "What is the sum of 265 and 345?"
    },
    {
        "prompt": "Greet shrek"
    },
    {
        "prompt": "Greet john"
    },
    {
        "prompt": "Reverse the string 'hello'"
    },
    ...
]
```

`data/input/function_definitions.json` must contain the available functions your the system can call. Each function includes:

- Function name
- Argument names and types
- Return type
- Description

Example: `function_definitions.json`
```json
[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {
                "type": "number"
            },
            "b": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    },
    {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person by name.",
        "parameters": {
            "name": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    },
    {
        "name": "fn_reverse_string",
        "description": "Reverse a string and return the reversed result.",
        "parameters": {
            "s": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    },
    ...
]
```

### Output

The output is written as a JSON array to `data/output/function_calling_results.json`. Each object in the array contains exactly the following keys:

- prompt (string): The original natural-language request
- name (string): The name of the function to call
- parameters (object): All required arguments with the correct types

Example output:
```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0}
    },
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": {"s": "hello"}
    }
]
```

## Instructions

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

## Notes

### Algorithm explanation

My implementation of constrained decoding makes heavy use of regular expressions (regex) to restrict the LLM's output. For each potential token, a regex is used to check whether it fits the expected output. This way, at every step of the way, we choose the best token that still matches valid JSON formatting.

### Design decisions

In order to simplify the constrained decoding process, I implemented a very rudimentary state machine system. A manager class updates its internal state throughout the output process: whenever a state is fulfilled, it switches to the next state. While not the fastest or most elegant solution, it was the simplest way to update the regex used for each step of the process.

Here is an example of how it works:
- At first, JSON expects an opening curcly brace ('{'). The first state class forces the LLM to output a token that matches this character.
- Then, the project's format expects a 'prompt' key. This second state restricts the LLM's tokens to match this regex: `r'"prompt":\s?'`.
- ...
- At the very end, the last state constrains the LLM into outputting a closing curly brace ('}').

### Performance analysis

The solution I opted for is fairly slow as it forces the LLM to output tokens throughout the whole JSON. In order to optimize it a bit, I added a 'string' variable to each state that defined a fixed output value (for example, that forced the LLM to output exactly `"prompt":`). By tokenizing the string directly and adding it to the currently generated tokens, it allowed me to skip the selection process for many tokens and speed up generation considerably. While it is still not as fast as if I had only generated the required elements and created the JSON myself, I prefer this method as it forces the LLM the generate the whole JSON itself, within the imposed constraints.

This method is 100% reliable as it uses regular expressions to force the output to match the expected format.

As for accuracy, this is mostly linked to the amount of context. I gave the LLM a decent amount of context and instructions to maximize the accuracy, but it is still prone to mistakes. A good way to improve accuracy would be to let the LLM think a little before giving its answers, but this also decreases speed a lot, so I decided to remove it from the final program.

### Challenges faced

The main challenge I faced was figuring out how to implement a state machine and switch from one state to another with separate regexes. Since the regex for the `"parameters"` key must be updated after the LLM picked a function name, it is not possible to use a single set regex for the whole decoding process. A state machine must be used to generate and apply different regexes at different steps of the decoding processes, allowing regexes to be created throughout the generation, with values updated on the fly.

Another difficulty was implementing Pydantic classes throughout the project (as requested by the subject) despite it not being a good fit for the structure.

### Testing strategy

Validation was done through manual testing of different input prompts and functions as automated testing would be quite difficult to apply to this project.

### Example usage

#### Example 1

Prompt in `input/function_calling_tests.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  }
]
```
Answer in `output/function_calls.json`:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2,
            "b": 3
        }
    }
]
```

#### Example 2

Prompt in `input/function_calling_tests.json`:

```json
[
  {
    "prompt": "Replace all vowels in 'Programming is fun' with asterisks"
  }
]
```
Answer in `output/function_calls.json`:

```json
[
    {
        "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "Programming is fun",
            "regex": "([aeiouAEIOU])",
            "replacement": "*"
        }
    }
]
```

## Resources

- [A Guide to Structured Generation Using Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Building a simple State Machine in Python.](https://dev.to/karn/building-a-simple-state-machine-in-python)
- [Introducing JSON](https://www.json.org/json-en.html)
- [Regex101](https://regex101.com/)
- AI was used to
    - help understand basic LLM functionalities
    - explain how to implement constrained decoding
    - teach how to apply pydantic models to the project
