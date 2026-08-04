import re
import threading
from typing import TYPE_CHECKING

from faasr_agents.faasr.runtime.faasr_function_logger import FaaSrFunctionLogger, LogEvent
from faasr_agents.faasr.runtime.utils import get_s3_path
from faasr_agents.faasr.runtime.enums import FunctionStatus

if TYPE_CHECKING:
    from faasr_agents.faasr.runtime.s3_client import FaaSrS3Client
else:
    FaaSrS3Client = object


class FaaSrFunction:
    """
    Manages the execution status and monitoring of a single FaaSr function.

    This class is responsible for:
    - Tracking function execution status
    - Listening to logger events and updating status reactively
    - Tracking invocations from log analysis
    - Managing function completion and failure detection
    """

    failed_regex = re.compile(r"\[[\d\.]+?\] \[ERROR\]")
    invoked_regex = re.compile(
        r"(?<=\[scheduler.py\] GitHub Action: Successfully invoked: )[a-zA-Z][a-zA-Z0-9\-]+"
    )

    def __init__(
        self,
        *,
        function_name: str,
        workflow_name: str,
        invocation_folder: str,
        s3_client: FaaSrS3Client,
        stream_logs: bool = False,
        interval_seconds: int = 3,
        start_logger: bool = True,
    ):
        self.function_name = function_name
        self.workflow_name = workflow_name
        self.invocation_folder = invocation_folder
        self.s3_client = s3_client
        self.stream_logs = stream_logs
        self.interval_seconds = interval_seconds

        # Initialize function logger
        self._logger = FaaSrFunctionLogger(
            function_name=function_name,
            workflow_name=workflow_name,
            invocation_folder=invocation_folder,
            s3_client=s3_client,
            stream_logs=stream_logs,
            interval_seconds=interval_seconds,
        )

        # Status management
        self._status = FunctionStatus.PENDING
        self._invocations: set[str] | None = None

        # Thread safety
        self._lock = threading.Lock()

        # Register callbacks with logger
        self._logger.register_callback(self._on_log_event)

        if start_logger:
            self._logger.start()

    @property
    def status(self) -> FunctionStatus:
        """
        Get the current function status (thread-safe).

        Returns:
            FunctionStatus: The current status of the function.
        """
        with self._lock:
            return self._status

    @property
    def done_key(self) -> str:
        """
        Get the complete `.done` file S3 key.

        This replaces ranks with the expected format
        (e.g. "function(1)" -> "function.1.done").

        Returns:
            str: The S3 key for the done file.
        """
        s3_function_name = self.function_name.replace("(", ".").replace(")", "")
        return get_s3_path(
            f"{self.invocation_folder}/function_completions/{s3_function_name}.done"
        )

    @property
    def invocations(self) -> set[str] | None:
        """
        Get the invocations (thread-safe).

        Returns:
            set[str] | None: The invocations.
        """
        with self._lock:
            if self._invocations is None:
                return None
            return self._invocations.copy()

    def set_status(self, status: FunctionStatus) -> None:
        """
        Set the function status (thread-safe).

        Args:
            status: The new status to set.
        """
        with self._lock:
            self._status = status

    def start(self) -> None:
        """Start monitoring the function."""
        self._logger.start()

    def _on_log_event(self, event: LogEvent) -> None:
        """
        Handle log events from the function logger.

        Args:
            event: The log event that occurred
        """
        match event:
            case LogEvent.LOG_CREATED:
                self._handle_log_created()
            case LogEvent.LOG_UPDATED:
                self._handle_log_updated()
            case LogEvent.LOG_COMPLETE:
                self._handle_log_complete()

    def _handle_log_created(self) -> None:
        """Handle when logs are first created."""
        # Function is now running
        self.set_status(FunctionStatus.RUNNING)

    def _handle_log_updated(self) -> None:
        """Handle when logs are updated."""
        # Check for failures
        if self._check_for_failure():
            self.set_status(FunctionStatus.FAILED)
            self._logger.stop()
            self._logger.wait(timeout=2.0)
        # Check for completion
        elif self._check_for_completion():
            self.set_status(FunctionStatus.COMPLETED)
            self._logger.stop()
            self._logger.wait(timeout=2.0)

    def _handle_log_complete(self) -> None:
        """Handle when logs are complete."""
        # Extract invocations from logs
        self._extract_invocations()
        # Final status check
        if self._check_for_failure():
            self.set_status(FunctionStatus.FAILED)
        elif self._check_for_completion():
            self.set_status(FunctionStatus.COMPLETED)

    def _check_for_failure(self) -> bool:
        """
        Check if the logs contain an error-level log.

        Returns:
            bool: True if the logs contain an error-level log, False otherwise.
        """
        return self.failed_regex.search(self._logger.logs_content) is not None

    def _check_for_completion(self) -> bool:
        """
        Check if the function has completed by looking for the .done file.

        Returns:
            bool: True if the function has completed, False otherwise.
        """
        return self.s3_client.object_exists(self.done_key)

    def _extract_invocations(self) -> None:
        """
        Extract invocations from the logs (thread-safe).

        This pulls the names of all invoked functions from the logs (excluding ranks).
        """
        logs_content = self._logger.logs_content
        with self._lock:
            self._invocations = set(
                re.sub(r"^" + self.workflow_name + "-", "", invocation)
                for invocation in self.invoked_regex.findall(logs_content)
            )

    @property
    def logs(self) -> list[str]:
        """Get the logs as a list (thread-safe)."""
        return self._logger.logs

    @property
    def logs_content(self) -> str:
        """Get the logs as a string (thread-safe)."""
        return self._logger.logs_content

    @property
    def logs_complete(self) -> bool:
        """Get the logs complete flag (thread-safe)."""
        return self._logger.logs_complete

    @property
    def logs_started(self) -> bool:
        """Get the logs started flag (thread-safe)."""
        return self._logger.logs_started
