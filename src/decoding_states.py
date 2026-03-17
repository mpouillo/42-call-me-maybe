from src import State
from src import JSONState as js


class StartState(State):
    """Initial step when beginning constrained decoding"""

    def on_event(self, event):
        if event == js.AFTER_BRACE:
            return AfterBraceState()


class AfterBraceState(State):
    """Initial step when beginning constrained decoding"""

    def on_event(self, event):
        if event == js.INSIDE_KEY:
            return InsideKeyState()


class InsideKeyState(State):
    """Initial step when beginning constrained decoding"""

    def on_event(self, event):
        if event == js.AFTER_KEY:
            return AfterKeyState()


class AfterKeyState(State):
    """Initial step when beginning constrained decoding"""

    def on_event(self, event):
        if event == js.VALUE_START:
            return ValueStartState()


class ValueStartState(State):
    """Initial step when beginning constrained decoding"""

    def on_event(self, event):
        if event == js.VALUE_END:
            return StartState()
