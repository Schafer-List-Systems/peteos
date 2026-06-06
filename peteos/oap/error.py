"""OAP Error class - a returned value object, not an exception."""


class Error:
    """Represents a task-level failure where the agent worked but could not produce the desired result."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __repr__(self) -> str:
        return f"Error('{self.message}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Error):
            return False
        return self.message == other.message

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
