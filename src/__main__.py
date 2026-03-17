#!/usr/bin/env python3

import argparse
import sys
from src import GenerationDevice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions-definition",
        default="data/input/functions_definition.json",
        help="Path to functions definition file"
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

    parser.add_argument(
        "--prompt",
        default="What is 2+2? Answer in less than 20 words.",
        help="Prompt"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        device = GenerationDevice()
        print(device.prompt(args.prompt))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
