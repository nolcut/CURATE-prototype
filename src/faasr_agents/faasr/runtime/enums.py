from enum import Enum


class InvocationStatus(Enum):
    PENDING = "pending"
    INVOKED = "invoked"
    NOT_INVOKED = "not_invoked"


class FunctionStatus(Enum):
    """Function execution status"""

    PENDING = "pending"
    INVOKED = "invoked"
    NOT_INVOKED = "not_invoked"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
