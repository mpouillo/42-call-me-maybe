import json
import numpy
import regex
import sys
from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model


class ConstrainedDecoder:
    def __init__(self,
                 prompts: list[str],
                 definitions: list[dict[str, Any]],
                 model: Any | None = None) -> None:
        self.prompts = prompts
        self.definitions: list[dict[str, Any]] = definitions

        if model:
            self.model = model
        else:
            self.model = Small_LLM_Model()

        self.vocab = self.load_vocabulary()


    def load_vocabulary(self) -> dict[int, str]:
        try:
            with open(self.model.get_path_to_vocab_file(), "r") as f:
                data = json.load(f)
            return {v: k for k, v in data.items()}
        except (FileNotFoundError, PermissionError):
            sys.exit("Error while parsing model vocabulary")

    def get_context(self, user_prompt):
        return f"""
            ### Instruction:
            You are a function-calling assistant. Based on the user's input, you must select the appropriate function and extract the correct parameters.

            ### Available Functions:
            {json.dumps(self.definitions, indent=2)}

            ### Constraint Rules:
            - CRITICAL: The output must always be valid JSON.
            - Function names must be one of the ones defined above.
            - Parameters must match the types defined above.
            - Limit your thoughts to maximum 10 sentences (as defined per ending with a '.').

            ### User Input:
            {user_prompt}

            ### Assistant Response (JSON):
            """

    def get_best_token(self, manager, next_logits):
        sorted_tokens = numpy.argsort(next_logits)[::-1]

        for token_id in sorted_tokens:
            tid = int(token_id)

            if tid not in self.vocab:
                continue

            token_str = self.clean_token(self.vocab[token_id])
            candidate = manager.current_string + token_str

            if regex.fullmatch(manager.state.get_regex(), candidate, partial=True):
                return tid, token_str
        return 0

    def str_to_ids(self, string):
        return list(self.model.encode(string))[0].tolist()

    def clean_token(self, token):
        return token.replace('\u0120', ' ')

    def process_prompts(self) -> list[str]:
        final_output = []

        for i, prompt in enumerate(self.prompts, start=1):

            print(f"Processing prompt [{i}/{len(self.prompts)}]: '{prompt}'")

            manager = StateManager(prompt, self.definitions)
            context_ids = self.str_to_ids(self.get_context(prompt))
            gen_ids = []

            while manager.state != "done":

                logits = self.model.get_logits_from_input_ids(context_ids + gen_ids)
                token_id, token_str = self.get_best_token(manager, logits)

                manager.current_string += token_str
                manager.output_string += token_str
                gen_ids.append(token_id)

                manager.on_value(manager.current_string)

            final_output.append(manager.output_string)

        return final_output

class StateManager:
    """State machine representing the generation steps of an LLM"""
    def __init__(self,
                 prompt: str,
                 definitions: list[dict]):
        self.prompt = prompt
        self.definitions = definitions
        self.current_string = ""
        self.output_string = ""
        self.state = self.StartCurlyBracesState(self)

    def on_value(self, value):
        new_state = self.state.on_value(value)
        if new_state is not None:
            self.current_string = ""
            self.state = new_state

    def get_regex(self):
        return self.state.get_regex()

    class StartCurlyBracesState:
        def __init__(self, outer):
            self.outer = outer

        def is_valid_continuation(self, candidate: str) -> bool:
            return regex.fullmatch(self.get_regex(), candidate)

        def get_regex(self):
            return r'\{'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.ThoughtKeyState(self.outer)

    class ThoughtKeyState:
        def __init__(self, outer):
            self.outer = outer

        def is_valid_continuation(self, candidate: str) -> bool:
            return regex.fullmatch(self.get_regex(), candidate)

        def get_regex(self):
            return r'"thought":\s?'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.ThoughtValueState(self.outer)

    class ThoughtValueState:
        def __init__(self, outer):
            self.outer = outer
            self.prefixes = [
                "I will use",
                "Based on the user input,",
                "To answer this,"
            ]

        def is_valid_continuation(self, candidate: str) -> bool:
            return regex.fullmatch(self.get_regex(), candidate)

        def get_regex(self):
            return (fr'"(?:({"|".join(self.prefixes)})) '
                    + r'(?:(?:[^"\\]|\\["\\/bfnrt]|\\u[a-fA-F0-9]{4})*?\. ?){1,10}",\s?')

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.PromptKeyState(self.outer)

    class PromptKeyState:
        def __init__(self, outer):
            self.outer = outer

        def is_valid_continuation(self, candidate: str) -> bool:
            return regex.fullmatch(self.get_regex(), candidate)

        def get_regex(self):
            return r'"prompt":\s?'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.PromptValueState(self.outer)

    class PromptValueState:
        def __init__(self, outer):
            self.outer = outer
            self.escaped_prompt = json.dumps(self.outer.prompt)[1:-1]

        def is_valid_continuation(self, candidate: str) -> bool:
            return regex.fullmatch(self.get_regex(), candidate)

        def get_regex(self):
            return fr'"{regex.escape(self.escaped_prompt)}",\s?'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.NameKeyState(self.outer)

    class NameKeyState:
        def __init__(self, outer):
            self.outer = outer

        def is_valid_continuation(self, candidate: str) -> bool:
            return regex.fullmatch(self.get_regex(), candidate)

        def get_regex(self):
            return r'"name":\s'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.NameValueState(self.outer)

    class NameValueState:
        def __init__(self, outer):
            self.outer = outer

        def is_valid_continuation(self, candidate: str) -> bool:
            return regex.fullmatch(self.get_regex(), candidate)

        def get_regex(self):
            return fr'"({"|".join(f['name'] for f in self.outer.definitions)})",\s?'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.ParametersKeyState(self.outer)

    class ParametersKeyState:
        def __init__(self, outer):
            self.outer = outer

        def is_valid_continuation(self, candidate: str) -> bool:
            return regex.fullmatch(self.get_regex(), candidate)

        def get_regex(self):
            return r'"parameters":\s'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.ParametersValueState(self.outer)

    class ParametersValueState:
        def __init__(self, outer):
            self.outer = outer
            self.regex_value = self._init_regex()

        def _init_regex(self):
            func_name_regex = self.outer.NameValueState(self.outer).get_regex()
            name_match = regex.search(func_name_regex, self.outer.output_string)
            if not name_match:
                raise ValueError("Could not find selected function name in output")
            selected_func = name_match.group(1)

            func_def = next(f for f in self.outer.definitions if f["name"] == selected_func)
            params = list(func_def["parameters"].items())
            total_regex = ""

            for i, (p_name, p_info) in enumerate(params):
                if p_info["type"] == "number":
                    p_regex = r'[+-]?([0-9]*[.])?[0-9]+'
                else:
                    p_regex = r'"(?:[^"\\]|\\["\\/bfnrt]|\\u[a-fA-F0-9]{4})*"'

                prefix = r'\{' if i == 0 else ', '
                total_regex += prefix + fr'"{p_name}":\s' + p_regex

            return total_regex + r'\}'

        def get_regex(self):
            return self.regex_value

        def is_valid_continuation(self, candidate: str) -> bool:
            return regex.fullmatch(self.get_regex(), candidate)

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.EndCurlyBracesState(self.outer)

    class EndCurlyBracesState:
        def __init__(self, outer):
            self.outer = outer

        def get_regex(self):
            return r'\}'

        def is_valid_continuation(self, candidate: str) -> bool:
            return regex.fullmatch(self.get_regex(), candidate)

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return "done"
