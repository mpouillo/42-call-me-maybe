import json
import numpy    # type: ignore
import regex    # type: ignore
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

    def build_mask(self, reg_exp, current_text, next_logits):
        mask = numpy.full(len(next_logits), -float('inf'))
        for token_id in range(len(next_logits)):    # Slow, to fix
            if token_id not in self.vocab:
                continue
            token_str = self.vocab[token_id]
            clean_token = self.clean_token(token_str)
            candidate = current_text + clean_token

            if regex.fullmatch(reg_exp, candidate, partial=True):
                mask[token_id] = 0

        return mask

    def get_context(self, user_prompt):
        return f"""
            ### Instruction:
            You are a function-calling assistant. Based on the user's input, you must select the appropriate function and extract the correct parameters.

            ### Available Functions:
            {json.dumps(self.definitions, indent=2)}

            ### Constraint Rules:
            - Function names must be one of the ones defined above.
            - Parameters must match the types defined above.
            - The output must be valid JSON.

            ### User Input:
            {user_prompt}

            ### Assistant Response (JSON):
            """

    def str_to_ids(self, string):
        return list(self.model.encode(string))[0].tolist()

    def clean_token(self, token):
        return token.replace('\u0120', ' ')

    def process_prompts(self) -> list[str]:
        final_output = []

        for i, prompt in enumerate(self.prompts, start=1):

            print(f"Processing prompt [{i}/{len(self.prompts)}]: '{prompt}'")

            manager = SequenceManager(prompt, self.definitions)
            context_ids = self.str_to_ids(self.get_context(prompt))
            gen_ids = []

            while manager.state != "done":

                logits = self.model.get_logits_from_input_ids(context_ids + gen_ids)
                mask = self.build_mask(manager.get_regex(), manager.current_string, logits)

                next_token_id = numpy.argmax(logits + mask)
                if next_token_id == self.EOS_TOKEN:
                    continue

                token_str = self.clean_token(self.vocab[next_token_id])
                manager.current_string += token_str
                manager.output_string += token_str
                gen_ids.append(next_token_id)

                manager.on_value(manager.current_string)

            final_output.append(manager.output_string)

        return final_output

class SequenceManager:
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

        def get_regex(self):
            return r'\{'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.ThoughtKeyState(self.outer)

    class ThoughtKeyState:
        def __init__(self, outer):
            self.outer = outer

        def get_regex(self):
            return r'"thought":\s'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.ThoughtValueState(self.outer)

    class ThoughtValueState:
        def __init__(self, outer):
            self.outer = outer

        def get_regex(self):
            return r'"(I will use|Based on the user input,|To answer this,) [^"]{0,999}",\s'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.PromptKeyState(self.outer)

    class PromptKeyState:
        def __init__(self, outer):
            self.outer = outer

        def get_regex(self):
            return r'"prompt":\s'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.PromptValueState(self.outer)

    class PromptValueState:
        def __init__(self, outer):
            self.outer = outer

        def get_regex(self):
            escaped_prompt = json.dumps(self.outer.prompt)[1:-1]
            return fr'"{regex.escape(escaped_prompt)}",\s?'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.NameKeyState(self.outer)

    class NameKeyState:
        def __init__(self, outer):
            self.outer = outer

        def get_regex(self):
            return r'"name":\s'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.NameValueState(self.outer)

    class NameValueState:
        def __init__(self, outer):
            self.outer = outer

        def get_regex(self):
            return fr'"({"|".join(f['name'] for f in self.outer.definitions)})",\s?'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.ParametersKeyState(self.outer)

    class ParametersKeyState:
        def __init__(self, outer):
            self.outer = outer

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
                    p_regex = r'"(?:[^"\\]|\\.)*"'

                prefix = r'\{' if i == 0 else ', '
                total_regex += prefix + fr'"{p_name}":\s' + p_regex

            return total_regex + r'\}'

        def get_regex(self):
            return self.regex_value

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return self.outer.EndCurlyBracesState(self.outer)

    class EndCurlyBracesState:
        def __init__(self, outer):
            self.outer = outer

        def get_regex(self):
            return r'\}'

        def on_value(self, value):
            if regex.fullmatch(self.get_regex(), value):
                return "done"
