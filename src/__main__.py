#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
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


def validate_output_path(output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_dir():
        print(f"Error: output file is a directory", file=sys.stderr)
        sys.exit(1)

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
        print(f"Error parsing input file: {e}", file=sys.stderr)
        sys.exit(1)


def parse_definitions(definitions_file: str) -> list[dict[str, Any]]:
    try:
        with open(definitions_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error parsing definitions file: {e}", file=sys.stderr)
        sys.exit(1)


def print_output_to_file(model_answers: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        processed_answers = [json.loads(answer) for answer in model_answers]
        for answer in processed_answers:
            answer.pop("thought")
    except json.JSONDecodeError as e:
        print(f"Error encoding model output to JSON objects: {e}")
        sys.exit(1)

    if path.is_dir():
        print(f"Error: output path is a directory", file=sys.stderr)
        sys.exit(1)

    try:
        with path.open("w") as f:
            json.dump(processed_answers, f, indent=4)
            print(f"Done! Output written to '{path}'")
    except Exception as e:
        print(f"Error writing LLM output to file: {e}", file=sys.stderr)
        sys.exit(1)

def main() -> None:
    args = parse_args()
    prompts = parse_prompts(args.input)
    definitions = parse_definitions(args.definitions)
    validate_output_path(args.output)

    decoder = ConstrainedDecoder(prompts, definitions)
    model_answers = decoder.process_prompts()
    print_output_to_file(model_answers, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        os._exit(1)
