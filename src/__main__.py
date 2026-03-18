#!/usr/bin/env python3

import argparse
import json
import sys
from src import ConstrainedDecoder
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions-definition",
        default="data/input/functions_definition.json",
        help="Path to functions definition file",
        dest="definitions"
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to input prompts file"
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path to save results"
    )

    return parser.parse_args()


def parse_prompts(input_file: str) -> list[str]:
    prompts: list[str] = []
    try:
        with open(input_file, 'r') as f:
            for prompt in json.load(f):
                p = prompt.get("prompt")
                if p:
                    prompts.append(p)
        return prompts
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def parse_definitions(definitions_file: str) -> list[dict[str, Any]]:
    try:
        with open(definitions_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    prompts: list[str] = parse_prompts(args.input)
    definitions: list[dict[str, Any]] = parse_definitions(args.definitions)

    decoder = ConstrainedDecoder(prompts, definitions)
    output = decoder.process_prompts()
    from pprint import pprint
    pprint(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
