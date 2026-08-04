from faasr_agents.faasr.runtime.enums import FunctionStatus

STATUS_STYLE: dict[FunctionStatus, tuple[str, str]] = {
    FunctionStatus.PENDING:     ("dim",        "PENDING"),
    FunctionStatus.INVOKED:     ("yellow",     "INVOKED"),
    FunctionStatus.RUNNING:     ("blue bold",  "RUNNING"),
    FunctionStatus.COMPLETED:   ("green bold", "COMPLETED"),
    FunctionStatus.FAILED:      ("red bold",   "FAILED"),
    FunctionStatus.SKIPPED:     ("dim strike", "SKIPPED"),
    FunctionStatus.TIMEOUT:     ("red",        "TIMEOUT"),
    FunctionStatus.NOT_INVOKED: ("dim",        "NOT INVOKED"),
}
