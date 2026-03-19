import json
import numpy
import regex    # type: ignore
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
        self.OUTPUT_LIMIT = 200
        self.EOS_TOKEN = self.get_eos_token()

    def get_eos_token(self) -> int:
        try:
            with open(self.model.get_path_to_tokenizer_file(), "r") as f:
                data = json.load(f)
            return data["added_tokens"][0]["id"]
        except (FileNotFoundError, PermissionError):
            sys.exit("Error while parsing model vocabulary")

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

    def build_mask(self, reg_exp, current_text, next_logits):
        mask = numpy.full(len(self.vocab), -float('inf'))
        for token_id in next_logits:
            token_str = self.vocab[token_id]
            clean_token = self.clean_token(token_str)
            candidate = current_text + clean_token

            if regex.fullmatch(reg_exp, candidate, partial=True):
                mask[token_id] = 0

        return mask

    def str_to_ids(self, string):
        return list(self.model.encode(string))[0].tolist()

    def clean_token(self, token):
        return token.replace('\u0120', ' ')

    def process_prompts(self) -> list[str]:
        final_output = []
        self.prompts = self.prompts[:2]
        for i, prompt in enumerate(self.prompts, start=1):
            print(f"Processing prompt [{i}/{len(self.prompts)}]")
            manager = SequenceManager(prompt, self.definitions)
            current_ids = self.str_to_ids(prompt)

            while manager.current_index < len(manager.parts):
                part = manager.parts[manager.current_index]

                if part.static_prefix and not manager.output_string.endswith(part.static_prefix):
                    manager.output_string += part.static_prefix
                    prefix_ids = self.str_to_ids(part.static_prefix)
                    current_ids.extend(prefix_ids)
                    continue

                if part.is_dynamic:
                    func_match = regex.search(manager.func_name_regex, manager.output_string)
                    if not func_match:
                        raise ValueError("Could not find selected function name in output")

                    selected_func = func_match.group(0)
                    func_def = next(f for f in self.definitions if f["name"] == selected_func)
                    new_parts = []
                    params = list(func_def["parameters"].items())

                    for i, (p_name, p_info) in enumerate(params):
                        if p_info["type"] == "number":
                            p_regex = r'[+-]?([0-9]*[.])?[0-9]+'
                        else:
                            p_regex = r'"[^"]*"'

                        prefix = '{' if i == 0 else ', '
                        new_parts.append(ExpectedPart(f'{prefix}"{p_name}": ', p_regex))

                    new_parts.append(ExpectedPart('}', ""))
                    manager.parts[manager.current_index:manager.current_index + 1] = new_parts

                    continue

                logits = self.model.get_logits_from_input_ids(current_ids)
                mask = self.build_mask(part.constraint_regex, manager.output_string, logits)

                next_token_id = numpy.argmax(logits + mask)

                if next_token_id == self.EOS_TOKEN:
                    manager.current_index += 1
                    continue

                token_str = self.clean_token(self.vocab[next_token_id])
                manager.output_string += token_str
                current_ids.append(next_token_id)

                if regex.fullmatch(part.constraint_regex, manager.extract_value(part)):
                    manager.current_index += 1

            final_output.append(manager.output_string)

        return final_output


class ExpectedPart:
    def __init__(self,
                 static_prefix: str,
                 constraint_regex: str,
                 is_dynamic: bool = False):
        self.static_prefix = static_prefix
        self.constraint_regex = constraint_regex
        self.is_dynamic = is_dynamic


class SequenceManager:
    def __init__(self,
                 prompt: str,
                 functions: list[dict]):
        self.prompt = prompt
        self.functions = functions
        self.func_name_regex = "(" + "|".join(f['name'] for f in functions) + ")"
        self.parts = [
            ExpectedPart('{"prompt": "',
                         regex.escape(prompt)),
            ExpectedPart('", "name": "',
                         self.func_name_regex),
            ExpectedPart('", "parameters": ',
                         "",
                         is_dynamic=True),
            ExpectedPart('}', "")
        ]
        self.current_index = 0
        self.output_string = ""

    def extract_value(self, part: ExpectedPart) -> str:
        prefix_end = self.output_string.rfind(part.static_prefix) + len(part.static_prefix)
        return self.output_string[prefix_end:]
