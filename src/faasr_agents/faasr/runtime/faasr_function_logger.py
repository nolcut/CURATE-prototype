import logging
import re
import threading
import time
from enum import Enum
from typing import Callable

from faasr_agents.faasr.runtime.s3_client import FaaSrS3Client
from faasr_agents.faasr.runtime.utils import get_s3_path


class LogEvent(Enum):
    """Events that can be triggered by the FaaSrFunctionLogger"""

    LOG_CREATED = "log_created"
    LOG_UPDATED = "log_updated"
    LOG_COMPLETE = "log_complete"


class FaaSrFunctionLogger:
    """
    Handles log monitoring and fetching for a single FaaSr function.

    This class is responsible for:
    - Monitoring logs on S3 for a specific function
    - Fetching and storing log content
    - Providing event-driven callbacks for log updates
    - Tracking log completion status

    Args:
        function_name: The name of the function.
        workflow_name: The name of the workflow.
        invocation_folder: The folder where the logs are stored.
        s3_client: The S3 client to use.
        stream_logs: Whether to stream the logs to the console.
        interval_seconds: The interval in seconds to check for new logs.
    """

    entry_regex = re.compile(r"\[[\d\.]+?\][\s\S]+?(?=\n\[)|\[[\d\.]+?\][\s\S]+?$")

    def __init__(
        self,
        *,
        function_name: str,
        workflow_name: str,
        invocation_folder: str,
        s3_client: FaaSrS3Client,
        stream_logs: bool = False,
        interval_seconds: int = 3,
    ):
        self.function_name = function_name
        self.workflow_name = workflow_name
        self.invocation_folder = invocation_folder
        self.s3_client = s3_client
        self.stream_logs = stream_logs
        self.interval_seconds = interval_seconds

        # Setup logger
        self.logger_name = function_name
        self.logger = self._setup_logger()

        # Log storage
        self._logs: list[str] = []

        # State tracking
        self._logs_started = False
        self._logs_complete = False

        # Thread management
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_requested = False

        # Event callbacks
        self._callbacks: list[Callable[[LogEvent], None]] = []

    def _setup_logger(self) -> logging.Logger:
        """
        Initialize the FaaSrFunctionLogger logger. Log outputs include the function name.

        Returns:
            logging.Logger: The logger for the FaaSrFunctionLogger.
        """
        logger = logging.getLogger(self.logger_name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f"[{self.logger_name}] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    @property
    def logs(self) -> list[str]:
        """
        Get the logs as a list (thread-safe).

        Returns:
            list[str]: The logs.
        """
        with self._lock:
            return self._logs.copy()

    @property
    def logs_content(self) -> str:
        """
        Get the logs as a string (thread-safe).

        Returns:
            str: The logs content.
        """
        with self._lock:
            return "\n".join(self._logs)

    @property
    def logs_key(self) -> str:
        """
        Get the complete logs S3 key.

        Returns:
            str: The S3 key for the logs.
        """
        return get_s3_path(f"{self.invocation_folder}/{self.function_name}.txt")

    @property
    def logs_started(self) -> bool:
        """
        Get the logs started flag (thread-safe).

        Returns:
            bool: True if logs have started appearing on S3.
        """
        with self._lock:
            return self._logs_started

    @property
    def logs_complete(self) -> bool:
        """
        Get the logs complete flag (thread-safe).

        This is True when the `.done` file exists and no new logs were fetched after a
        monitoring cycle.

        Returns:
            bool: True if logs are complete, False otherwise.
        """
        with self._lock:
            return self._logs_complete

    @property
    def stop_requested(self) -> bool:
        """
        Get the stop flag (thread-safe).

        Returns:
            bool: True if the logger's stop flag is set, False otherwise.
        """
        with self._lock:
            return self._stop_requested

    def register_callback(self, callback: Callable[[LogEvent], None]) -> None:
        """
        Register a callback to be called when log events occur.

        Args:
            callback: Function to call with LogEvent parameter
        """
        with self._lock:
            self._callbacks.append(callback)

    def _call_callbacks(self, event: LogEvent) -> None:
        """
        Call all registered callbacks with the given event.

        Args:
            event: The log event that occurred
        """
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in callback: {e}")

    def _update_logs(self, new_logs: list[str]) -> None:
        """
        Update the logs (thread-safe).

        Args:
            new_logs: The new logs to add to the logs.
        """
        with self._lock:
            self._logs += new_logs

    def _set_logs_started(self) -> None:
        """Set the logs started flag to True (thread-safe)."""
        with self._lock:
            self._logs_started = True

    def _set_logs_complete(self) -> None:
        """Set the logs complete flag to True (thread-safe)."""
        with self._lock:
            self._logs_complete = True

    def _check_for_logs(self) -> bool:
        """
        Check if the logs exist on S3.

        Returns:
            bool: True if logs exist on S3, False otherwise.
        """
        return self.s3_client.object_exists(self.logs_key)

    def _get_logs(self) -> list[str]:
        """
        Get the logs from S3.

        Returns:
            list[str]: The logs.
        """
        logs = self.s3_client.get_object(self.logs_key).strip()
        return self.entry_regex.findall(logs)

    def start(self) -> None:
        """Start the FaaSrFunctionLogger."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the FaaSrFunctionLogger."""
        with self._lock:
            self._stop_requested = True

    def wait(self, timeout: float | None = None) -> None:
        """
        Wait for the logger thread to finish.

        Args:
            timeout: Maximum time to wait in seconds. If None, wait indefinitely.
        """
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        """
        Run the main logging loop.

        Until logs are complete, this:
        - Fetches new logs from S3.
        - Updates the logs.
        - Triggers appropriate events.
        - Sets the logs complete flag if no new logs were fetched after a monitoring
          cycle.

        If `stream_logs` is True, this will also log the logs to the console.
        """
        while not self.logs_complete:
            # Handle stop requested when logs have not started
            if not self.logs_started and self.stop_requested:
                self._set_logs_complete()
                self._call_callbacks(LogEvent.LOG_COMPLETE)
                break

            if not self.logs_started and self._check_for_logs():
                self._set_logs_started()
                self._call_callbacks(LogEvent.LOG_CREATED)
            elif self.logs_started:
                log_content = self._get_logs()
                num_existing_logs = len(self.logs)
                new_logs = log_content[num_existing_logs:]

                if new_logs:
                    self._update_logs(new_logs)
                    self._call_callbacks(LogEvent.LOG_UPDATED)

                    if self.stream_logs:
                        for log in new_logs:
                            self.logger.info(log)

                # Check if logs are complete (no new logs after a cycle)
                if self.stop_requested and not new_logs:
                    self._set_logs_complete()
                    self._call_callbacks(LogEvent.LOG_COMPLETE)

            time.sleep(self.interval_seconds)
