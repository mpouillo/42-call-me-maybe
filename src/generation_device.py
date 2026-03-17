import json
import numpy
import sys
from typing import Any


class GenerationDevice(object):
    """State machine representing the generation steps of an LLM"""
    def __init__(self):
        from src import Small_LLM_Model
        self.model = Small_LLM_Model()
        self.tokenizer = self.load_tokenizer_from_file()
        self.OUTPUT_LIMIT = 50
        self.EOS_ID = self.tokenizer["added_tokens"][0].get("content")
        if not self.EOS_ID:
            raise ValueError("No EOF token found")

    def load_tokenizer_from_file(self) -> dict[str, int]:
        try:
            with open(self.model.get_path_to_tokenizer_file(), "r") as f:
                return json.load(f)
        except (FileNotFoundError, PermissionError):
            sys.exit("Error while trying to access tokenizer JSON file")

    def on_event(self, event):
        self.state = self.state.on_event(event)

    def prompt(self, prompt: str) -> str:
        input_ids = list(self.model.encode(prompt))[0].tolist()
        generated_ids: list[Any] = []
        print(prompt, "->", input_ids)

        while len(generated_ids) < self.OUTPUT_LIMIT:
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token_id = numpy.argmax(logits)

            print(next_token_id)
            if next_token_id == self.EOS_ID:
                break

            generated_ids.append(next_token_id)
            input_ids.append(next_token_id)

        output = self.model.decode(generated_ids)
        return output
