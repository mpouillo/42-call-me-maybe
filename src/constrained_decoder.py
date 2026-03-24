import json
import numpy
import regex
import sys

from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, List, Dict, Optional, Tuple

from .manager import StateManager


class ConstrainedDecoder(BaseModel):
    """
    Decode LLM output using constrained generation to ensure valid JSON output.

    Attributes:
        prompts: A list of natural language prompts to process.
        definitions: A list of dictionaries defining the available functions.
        model: The LLM model instance used for generation.
        vocab: A dictionary mapping token IDs to their string representations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompts: List[str]
    definitions: List[Dict[str, Any]]
    model_name: Optional[str] = "Qwen/Qwen3-0.6B"
    model: Any = Field(default_factory=Small_LLM_Model)
    vocab: Dict[int, str] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the vocabulary after Pydantic validates the inputs."""
        self.model = Small_LLM_Model(self.model_name)
        self.vocab = self.load_vocabulary()

    def load_vocabulary(self) -> Dict[int, str]:
        """Load the model vocabulary from the JSON file."""
        try:
            with open(self.model.get_path_to_vocab_file(), "r") as f:
                data = json.load(f)
            return {int(v): k for k, v in data.items()}
        except (FileNotFoundError, PermissionError):
            print("Error while parsing model vocabulary", file=sys.stderr)
            sys.exit(1)

    def get_context(self, user_prompt):
        """
        Generate the context string with
        instructions and function schemas.
        """
        return f"""\
            ### Instruction:
            You are a function-calling assistant. \
Based on the user's input, you must select the \
appropriate function and extract the correct parameters.

            ### Available Functions:
            {json.dumps(self.definitions, indent=2)}

            ### Constraint Rules:
            - CRITICAL: The output must always be valid JSON.
            - Function names must be one of the ones defined above.
            - Parameters must match the types defined above.
            - Limit your thoughts to maximum 10 sentences \
(as defined per ending with a '.').

            ### User Input:
            {user_prompt}

            ### Assistant Response (JSON):
            """

    def get_best_token(self, manager: 'StateManager',
                       next_logits: numpy.ndarray) -> Tuple[int, str]:
        """
        Find the most probable token that
        satisfies the current regex constraints.
        """
        sorted_tokens = numpy.argsort(next_logits)[::-1]

        for token_id in sorted_tokens:
            tid = int(token_id)

            if tid not in self.vocab:
                continue

            token_str = self.clean_token(self.vocab[token_id])
            candidate = manager.current_string + token_str

            if regex.fullmatch(manager.get_regex(), candidate, partial=True):
                return tid, token_str
        return 0, ""

    def str_to_ids(self, string: str) -> List[int]:
        """Encode a string into a list of token IDs."""
        return list(self.model.encode(string))[0].tolist()

    def clean_token(self, token: str) -> str:
        """Clean special characters from the token string."""
        return token.replace('\u0120', ' ')

    def process_prompts(self) -> List[str]:
        """Process all prompts and return the generated JSON strings."""
        final_output: List[str] = []

        for i, prompt in enumerate(self.prompts, start=1):

            print(f"Processing prompt [{i}/{len(self.prompts)}]: \"{prompt}\"")

            manager = StateManager(prompt=prompt, definitions=self.definitions)
            context_ids = self.str_to_ids(self.get_context(prompt))
            gen_ids: List[int] = []

            while manager.state != "done":
                if manager.is_solid_state():
                    string = manager.state.string
                    gen_ids.extend(self.str_to_ids(string))
                    manager.on_value(string)
                    continue

                logits = self.model.get_logits_from_input_ids(
                    context_ids + gen_ids
                )
                token_id, token_str = self.get_best_token(manager, logits)

                gen_ids.append(token_id)
                manager.on_value(token_str)

            final_output.append(manager.output_string)
            print("\n")

        return final_output
