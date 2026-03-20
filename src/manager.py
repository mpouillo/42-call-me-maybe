from pydantic import BaseModel, ConfigDict
from typing import Any, List, Dict


class StateManager(BaseModel):
    """
    Manages the state machine for generating constrained JSON output.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    definitions: List[Dict[str, Any]]
    current_string: str = ""
    output_string: str = ""
    state: Any = None

    def model_post_init(self, __context: Any) -> None:
        """Initializes the starting state after Pydantic validation."""
        from .states import OpeningCurlyBracesState
        self.state = OpeningCurlyBracesState(manager=self)

    def get_regex(self) -> str:
        """Retrieves the regex pattern for the current state."""
        return self.state.get_regex()

    def on_value(self, value: str) -> None:
        """Passes the current value to the state and updates the state if necessary."""
        new_state = self.state.on_value(value)
        if new_state is not None:
            self.current_string = ""
            self.state = new_state


StateManager.model_rebuild()
