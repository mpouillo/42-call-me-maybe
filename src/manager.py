from pydantic import BaseModel, ConfigDict
from typing import Any, List, Dict


class StateManager(BaseModel):
    """
    Manage the state machine for generating constrained JSON output.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    definitions: List[Dict[str, Any]]
    current_string: str = ""
    output_string: str = ""
    state: Any = None

    def model_post_init(self, __context: Any) -> None:
        """Initialize the starting state after Pydantic validation."""
        from .states import OpeningCurlyBracesState
        self.state = OpeningCurlyBracesState(manager=self)

    def is_solid_state(self) -> bool:
        return (
            self.state is not None
            and hasattr(self.state, "string")
            and self.state.string
        )

    def get_regex(self) -> Any:
        """Retrieve the regex pattern for the current state."""
        return self.state.get_regex()

    def on_value(self, value: str) -> None:
        """
        Append the current value to the strings
        and pass the current value to the state
        and update the state if necessary.
        """
        self.current_string += value
        self.output_string += value
        print(value, end="", flush=True)
        new_state = self.state.on_value(self.current_string)

        if new_state is not None:
            self.current_string = ""
            self.state = new_state


StateManager.model_rebuild()
