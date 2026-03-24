import json
import regex

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, List, Optional

from .manager import StateManager


class BaseState(BaseModel):
    """Base state class that all states inherit from"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    manager: "StateManager"
    string: Optional[str] = None

    def get_regex(self) -> str:
        raise NotImplementedError

    def on_value(self, value: str) -> Optional["BaseState"]:
        raise NotImplementedError


class OpeningCurlyBracesState(BaseState):
    """State checking for the opening curly brace of the JSON object."""

    string: str = "{"

    def get_regex(self) -> str:
        return regex.escape("{")

    def on_value(self, value: str) -> Optional['PromptKeyState']:
        if regex.fullmatch(self.get_regex(), value):
            return PromptKeyState(manager=self.manager)
        return None


class ThoughtKeyState(BaseState):
    """State checking for the 'thought' JSON key."""

    string: str = "\"thought\": "

    def get_regex(self):
        return r'"thought":\s?'

    def on_value(self, value: str) -> Optional['ThoughtValueState']:
        if regex.fullmatch(self.get_regex(), value):
            return ThoughtValueState(manager=self.manager)
        return None


class ThoughtValueState(BaseState):
    """State checking for the 'thought' JSON value."""

    prefixes: List[str] = Field(default_factory=lambda: [
        "I will use",
        "Based on the user input,",
        "To answer this,"
    ])

    def get_regex(self):
        prefixes = "|".join(regex.escape(p) for p in self.prefixes)
        return (
            fr'"(?:({prefixes})) '
            r'(?:(?:[^"\\]|\\["\\/bfnrt]|'
            r'\\u[a-fA-F0-9]{4})*?\. ?){1,10}",\s?'
        )

    def on_value(self, value: str) -> Optional['PromptKeyState']:
        if regex.fullmatch(self.get_regex(), value):
            return PromptKeyState(manager=self.manager)
        return None


class PromptKeyState(BaseState):
    """State checking for the 'prompt' JSON key."""

    string: str = "\"prompt\": "

    def get_regex(self):
        return r'"prompt":\s?'

    def on_value(self, value: str) -> Optional['PromptValueState']:
        if regex.fullmatch(self.get_regex(), value):
            return PromptValueState(manager=self.manager)
        return None


class PromptValueState(BaseState):
    """State checking for the 'prompt' JSON value."""

    escaped_prompt: str = ""

    def model_post_init(self, __context: Any) -> None:
        self.escaped_prompt = json.dumps(self.manager.prompt)[1:-1]
        self.string = f"\"{self.escaped_prompt}\", "

    def get_regex(self):
        return fr'"{regex.escape(self.escaped_prompt)}",\s?'

    def on_value(self, value: str) -> Optional['NameKeyState']:
        if regex.fullmatch(self.get_regex(), value):
            return NameKeyState(manager=self.manager)
        return None


class NameKeyState(BaseState):
    """State checking for the 'name' JSON key."""

    string: str = "\"name\": "

    def get_regex(self):
        return r'"name":\s?'

    def on_value(self, value: str) -> Optional['NameValueState']:
        if regex.fullmatch(self.get_regex(), value):
            return NameValueState(manager=self.manager)
        return None


class NameValueState(BaseState):
    """State checking for the 'name' JSON value."""

    def get_regex(self):
        name_patterns = "|".join(f["name"] for f in self.manager.definitions)
        return fr'"({name_patterns})",\s?'

    def on_value(self, value: str) -> Optional['ParametersKeyState']:
        if regex.fullmatch(self.get_regex(), value):
            return ParametersKeyState(manager=self.manager)
        return None


class ParametersKeyState(BaseState):
    """State checking for the 'parameters' JSON key."""

    string: str = "\"parameters\": "

    def get_regex(self):
        return r'"parameters":\s?'

    def on_value(self, value: str) -> Optional['ParametersValueState']:
        if regex.fullmatch(self.get_regex(), value):
            return ParametersValueState(manager=self.manager)
        return None


class ParametersValueState(BaseState):
    """State checking for the 'parameters' JSON value."""

    regex_value: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.regex_value:
            self.regex_value = self._init_regex()

    def _init_regex(self):
        name_state = NameValueState(manager=self.manager)
        name_match = regex.search(name_state.get_regex(),
                                  self.manager.output_string)
        if not name_match:
            raise ValueError("Could not find selected function name in output")
        selected_func = name_match.group(1)

        func_def = next(
            f for f in self.manager.definitions if f["name"] == selected_func
        )
        params = list(func_def["parameters"].items())
        total_regex = ""

        for i, (p_name, p_info) in enumerate(params):
            if p_info["type"] == "number":
                p_re = r'[+-]?([0-9]{0,999}[.])?[0-9]{1,999}'
            else:
                p_re = r'"(?:[^"\\]|\\["\\/bfnrt]|\\u[a-fA-F0-9]{4}){0,999}"'

            prefix = r'\{' if i == 0 else ', '
            total_regex += prefix + fr'"{p_name}":\s' + p_re

        return total_regex + r'\}'

    def get_regex(self):
        return self.regex_value

    def on_value(self, value: str) -> Optional['ClosingCurlyBracesState']:
        if regex.fullmatch(self.get_regex(), value):
            return ClosingCurlyBracesState(manager=self.manager)
        return None


class ClosingCurlyBracesState(BaseState):
    """State checking for the closing curly brace of the JSON object."""

    string: str = "}"

    def get_regex(self):
        return regex.escape("}")

    def on_value(self, value: str) -> Optional[str]:
        if regex.fullmatch(self.get_regex(), value):
            return "done"
        return None


BaseState.model_rebuild()
OpeningCurlyBracesState.model_rebuild()
ThoughtKeyState.model_rebuild()
ThoughtValueState.model_rebuild()
PromptKeyState.model_rebuild()
PromptValueState.model_rebuild()
NameKeyState.model_rebuild()
NameValueState.model_rebuild()
ParametersKeyState.model_rebuild()
ParametersValueState.model_rebuild()
ClosingCurlyBracesState.model_rebuild()
