#!/usr/bin/env python3

import argparse
import json
import os
import sys

from pathlib import Path
from typing import List, Any

from src import ConstrainedDecoder
from .models import PromptItem, FunctionDefinition, FunctionCallOutput


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


def validate_output_path(output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        print("Error: output file is a directory", file=sys.stderr)
        sys.exit(1)
    return path


def parse_prompts(input_file: str) -> List[str]:
    prompts: List[str] = []
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            for item in data:
                validated_item = PromptItem(**item)
                prompts.append(validated_item.prompt)
        return prompts
    except Exception as e:
        print(f"Error parsing input file: {e}", file=sys.stderr)
        sys.exit(1)


def parse_definitions(definitions_file: str) -> list[dict[str, Any]]:
    try:
        with open(definitions_file, 'r') as f:
            data = json.load(f)
            validated_defs = [FunctionDefinition(**d) for d in data]
            return [d.model_dump() for d in validated_defs]
    except Exception as e:
        print(f"Error parsing definitions file: {e}", file=sys.stderr)
        sys.exit(1)


def print_output_to_file(model_answers: str, output_path: Path) -> None:
    try:
        processed_answers = []
        for answer in model_answers:
            parsed_json = json.loads(answer)
            parsed_json.pop("thought", None)
            validated_answer = FunctionCallOutput(**parsed_json)
            processed_answers.append(validated_answer.model_dump())
    except json.JSONDecodeError as e:
        print(f"Error processing or validating model output: {e}")
        sys.exit(1)

    try:
        with output_path.open("w") as f:
            json.dump(processed_answers, f, indent=4)
            print(f"Done! Output written to '{output_path}'")
    except Exception as e:
        print(f"Error writing output to file: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    prompts = parse_prompts(args.input)
    definitions = parse_definitions(args.definitions)
    output_path = validate_output_path(args.output)

    decoder = ConstrainedDecoder(prompts=prompts, definitions=definitions)
    model_answers = decoder.process_prompts()
    print_output_to_file(model_answers, output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        os._exit(1)
