from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model


def run_generation(functions_definition_file: str,
                   input_file: str,
                   output_file: str) -> Any:
    model = Small_LLM_Model()
    print(model.encode("42"))
    return
