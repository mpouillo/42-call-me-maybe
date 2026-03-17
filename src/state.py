class State(object):
    """
    State object providing utility functions
    for the individual states within the state machine
    """

    def __init__(self):
        print("Processing current state:", str(self))

    def on_event(self, event):
        """Handle events that are delegated to this State."""
        pass

    def __repr__(self):
        """Leverage the __str__ method to describe the State."""
        return self.__str__()

    def __str__(self):
        """Return the name of the State."""
        return self.__class__.__name__
