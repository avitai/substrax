"""The exception classes of the fake Modal."""


class Error(Exception):
    """Every Modal error."""


class NotFoundError(Error):
    """A volume path or object that does not exist."""


class ExecutionError(Error):
    """A function call that raised."""
