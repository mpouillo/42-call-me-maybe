from enum import Enum


class JSONState(Enum):
    START = "start"
    AFTER_BRACE = "after_brace"
    INSIDE_KEY = "inside_key"
    AFTER_KEY = "after_key"
    VALUE_START = "value_start"
    VALUE_END = "value_end"
