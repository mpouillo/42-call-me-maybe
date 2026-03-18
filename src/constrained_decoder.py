import json
import numpy
import regex
import sys
from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model


class ConstrainedDecoder(object):
    """State machine representing the generation steps of an LLM"""
    def __init__(self, prompts: list[str], definitions: list[dict[str, Any]],
                 model: Any | None = None) -> None:
        self.prompts = prompts
        self.definitions: list[dict[str, Any]] = definitions

        if model:
            self.model = model
        else:
            self.model = Small_LLM_Model()

        self.stage = "prompt"
        self.vocab = self.load_vocabulary()
        self.OUTPUT_LIMIT = 50

    def load_vocabulary(self) -> dict[int, str]:
        try:
            with open(self.model.get_path_to_tokenizer_file(), "r") as f:
                data = json.load(f)

            vocab = data["model"]["vocab"].copy()
            for token in data.get("added_tokens", []):
                token_str = token["content"]
                token_id = token["id"]
                vocab[token_str] = token_id

            return {v: k for k, v in vocab.items()}

        except (FileNotFoundError, PermissionError):
            sys.exit("Error while parsing model vocabulary")

    def get_regex(self, prompt, current_text) -> regex:
        func_regexp = (
            '(' + "|".join([f.get("name", "") for f in self.definitions]) + ")"
        )

        match self.stage:
            case "prompt":
                return '^"prompt": ' + '"' + prompt + '"'
            case "name":
                return (
                    '"name": '
                    + '('
                    + "|".join([f.get("name", "") for f in self.definitions])
                    + ")"
                )
            case "parameters":
                selected_func = regex.match(func_regexp, current_text)
                for func_def in self.definitions:
                    params = None
                    if func_def.get("name") == selected_func:
                        params = func_def["parameters"].keys()
                param_regex = ", ".join([f"\"{p}\"" for p in params]) # \\\\\\\\\\\\\ WIP
                return '"parameters": ' + '{' + param_regex + '}'


    def mask_logits(self, prompt, current_ids, next_logits) -> list[float]:
        masked_logits = numpy.full(len(self.vocab), -float('inf'))
        current_text = self.model.decode(current_ids)

        for token_id, token_str in self.vocab.items():
            candidate = current_text + token_str.replace('\u0120', ' ')
            pattern = self.get_regex(prompt, current_text)
            if regex.fullmatch(pattern, candidate, partial=True):
                masked_logits[token_id] = next_logits[token_id]
            if regex.fullmatch(pattern, candidate):
                match self.stage:
                    case "prompt":
                        self.stage = "name"
                    case "name":
                        self.stage = "parameters"
                    case "parameters":
                        self.stage = "done"

        return list(masked_logits)

    def process_prompts(self) -> list[str]:
        output = []

        for i, prompt in enumerate(self.prompts):
            print(f"Processing prompt {i}/{len(self.prompts)}...")

            # Process prompts as IDs
            current_ids = list(self.model.encode(prompt))[0].tolist()
            input_lenght = len(current_ids)

            while len(current_ids) < self.OUTPUT_LIMIT + input_lenght:
                # Get logits from IDs
                logits = self.model.get_logits_from_input_ids(current_ids)
                # Constrain logits
                constrained_logits = self.mask_logits(prompt, current_ids, logits)
                # Get best ID from logits
                next_token_id = numpy.argmax(constrained_logits)

                # Append ID to existing IDs
                current_ids.append(next_token_id)

                if self.stage == "done":
                    break

            output.append(self.model.decode(current_ids[input_lenght:]))

        return output
