import logging
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from itertools import chain
from typing import Generator, Literal

from FaaSr_py import FaaSrPayload
from FaaSr_py.helpers.graph_functions import build_adjacency_graph
from FaaSr_py.helpers.s3_helper_functions import get_invocation_folder

from faasr_agents.faasr.runtime.faasr_function import FaaSrFunction
from faasr_agents.faasr.runtime.s3_client import FaaSrS3Client
from faasr_agents.faasr.runtime.utils import (
    completed,
    extract_function_name,
    failed,
    has_completed,
    has_final_state,
    invoked,
    not_invoked,
    pending,
    running,
)
from faasr_agents.faasr.runtime.enums import FunctionStatus, InvocationStatus

REQUIRED_ENV_VARS = [
    "S3_AccessKey",
    "S3_SecretKey",
    "GH_PAT",
    "GITHUB_REPOSITORY",
    "GITHUB_REF_NAME",
]

# Remove the existing logging handlers from the root logger
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
    handler.close()


class InitializationError(Exception):
    """Exception raised for WorkflowRunner initialization errors"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"Error initializing WorkflowRunner: {self.message}"


class StopMonitoring(Exception):
    """Exception raised to stop WorkflowRunner monitoring"""


class WorkflowRunner:
    """
    Runs a FaaSr workflow and monitors the execution of the functions.

    This class is responsible for:
    - Validating the environment
    - Setting up the logger
    - Building the adjacency graph
    - Initializing the function statuses
    - Initializing the S3 client
    - Setting up the signal handlers
    - Starting the monitoring thread
    - Shutting down the monitoring thread
    - Cleaning up the resources
    - Handling failures by waiting for active loggers to complete before cascading

    When a failure is detected, the runner waits for all loggers of functions that are
    INVOKED, RUNNING, or FAILED to complete before cascading the failure to pending
    functions. This ensures complete log information is available for debugging.

    Args:
        faasr_payload: The FaaSr payload.
        timeout: The timeout for the monitoring thread.
        check_interval: The interval for the monitoring thread.
        stream_logs: Whether to stream the logs to the console.

    Raises:
        InitializationError: If the environment is not valid.
    """

    logger_name = "WorkflowRunner"

    def __init__(
        self,
        *,
        faasr_payload: FaaSrPayload,
        timeout: int,
        check_interval: int,
        stream_logs: bool = False,
    ):
        self._validate_environment()
        self._faasr_payload = faasr_payload

        # Setup logging
        self.timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        self.logger = self._setup_logger()

        # Monitoring parameters
        self.timeout = timeout
        self.check_interval = check_interval
        self.last_change_time: float = time.time()
        self.seconds_since_last_change: float = 0.0

        # Thread management
        self._status_lock = threading.Lock()
        self._monitoring_thread = None
        self._monitoring_complete = False
        self._shutdown_requested = False
        self._cleanup_timeout = 30  # seconds to wait for graceful shutdown

        # Build adjacency graph for monitoring
        self.adj_graph, self.ranks = build_adjacency_graph(self._faasr_payload)
        self.reverse_adj_graph = self._build_reverse_adjacency_graph()

        # Initialize function statuses
        self.workflow_name = self._faasr_payload.get("WorkflowName")
        self.workflow_invoke = self._faasr_payload.get("FunctionInvoke")
        self.function_names = self._faasr_payload["ActionList"].keys()
        self._stream_logs = stream_logs
        self._functions: dict[str, FaaSrFunction] = {}
        self._prev_statuses: dict[str, FunctionStatus] = {}

        # Failure handling state
        self._failure_detected = False

        # Initialize S3 client for monitoring
        self.s3_client = FaaSrS3Client(
            workflow_data=self._faasr_payload,
            access_key=os.getenv("S3_AccessKey"),
            secret_key=os.getenv("S3_SecretKey"),
        )

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    @property
    def invocation_id(self) -> str:
        """Get the invocation ID of the workflow."""
        return self._faasr_payload["InvocationID"]

    ##########################
    # Initialization helpers #
    ##########################
    def _validate_environment(self) -> None:
        """
        Validate required environment variables.

        Raises:
            InitializationError: If any required environment variables are missing.
        """
        missing_env_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_env_vars:
            raise InitializationError(
                f"Missing required environment variables: {', '.join(missing_env_vars)}"
            )

    def _setup_logger(self) -> logging.Logger:
        """
        Initialize the WorkflowRunner logger.

        Returns:
            logging.Logger: The logger for the WorkflowRunner.
        """
        logger = logging.getLogger(self.logger_name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] [%(filename)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def _build_reverse_adjacency_graph(self) -> dict[str, set[str]]:
        """
        Initialize the reverse adjacency graph.

        Returns:
            dict[str, set[str]]: The reverse adjacency graph.
        """
        reverse_adj_graph = defaultdict(set)
        for invoker, invoked_functions in self.adj_graph.items():
            for function in invoked_functions:
                reverse_adj_graph[function].add(invoker)
        return reverse_adj_graph

    def _build_functions(self, stream_logs: bool) -> list[FaaSrFunction]:
        """
        Initialize the function statuses.

        Returns:
            list[FaaSrFunction]: The function instances.
        """
        functions: dict[str, FaaSrFunction] = {}
        for rank in chain(*(self._iter_ranks(name) for name in self.function_names)):
            function = FaaSrFunction(
                function_name=rank,
                workflow_name=self.workflow_name,
                invocation_folder=get_invocation_folder(self._faasr_payload),
                s3_client=self.s3_client,
                stream_logs=stream_logs,
            )
            if extract_function_name(rank) == self.workflow_invoke:
                function.set_status(FunctionStatus.INVOKED)
            functions[rank] = function
        return functions

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown on interruption."""

        def signal_handler(signum, frame):
            signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(signum))
            signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(signum))
            self.logger.info(
                f"Received signal {signum}, initiating graceful shutdown..."
            )
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    #########################
    # Thread-safe interface #
    #########################
    def get_function_statuses(self) -> dict[str, FunctionStatus]:
        """Get a copy of function statuses (thread-safe)."""
        with self._status_lock:
            return {
                function.function_name: function.status
                for function in self._functions.values()
            }

    def get_function_logs_content(self, function_name: str) -> str:
        """Get the logs content of a function."""
        with self._status_lock:
            return self._functions[function_name].logs_content

    @property
    def monitoring_complete(self) -> bool:
        """Check if monitoring is complete (thread-safe)."""
        with self._status_lock:
            return self._monitoring_complete

    @property
    def shutdown_requested(self) -> bool:
        """Check if a shutdown request has been made (thread-safe)."""
        with self._status_lock:
            return self._shutdown_requested

    def _set_monitoring_complete(self) -> None:
        """Set the monitoring complete status to True (thread-safe)."""
        with self._status_lock:
            self._monitoring_complete = True

    def _set_shutdown_requested(self) -> None:
        """Set the shutdown requested status to True (thread-safe)."""
        with self._status_lock:
            self._shutdown_requested = True

    @property
    def failure_detected(self) -> bool:
        """Get the failure detected status (thread-safe)."""
        with self._status_lock:
            return self._failure_detected

    def _set_failure_detected(self) -> None:
        """Set the failure detected status to True (thread-safe)."""
        with self._status_lock:
            self._failure_detected = True

    #######################
    # Workflow monitoring #
    #######################
    def _start_monitoring(self) -> None:
        """Start workflow monitoring."""
        self.logger.info(
            f"Monitoring workflow execution for functions: {', '.join(self.function_names)}"
        )

        self._reset_timer()

        while not self._did_timeout() and not self.shutdown_requested:
            try:
                self._monitor_workflow_execution()
            except StopMonitoring:
                break

            self._increment_timer()
            time.sleep(self.check_interval)

        self._finish_monitoring()

    def _monitor_workflow_execution(self):
        """Monitor the workflow execution."""
        for function in self._functions.values():
            if failed(function.status):
                if not self.failure_detected:
                    self._set_failure_detected()
                    self.logger.info(
                        f"Failure detected in function {function.function_name}. "
                        f"Waiting for active loggers to complete..."
                    )
            if self._prev_statuses[function.function_name] != function.status:
                self._log_status_change(function)
                self._prev_statuses[function.function_name] = function.status

        if self.failure_detected:
            active_functions = self._get_active_functions()
            if not active_functions:
                self.logger.info(
                    "All active functions have logs complete. Cascading failure..."
                )
                self._cascade_failure()
                raise StopMonitoring(
                    "Failure detected and all active loggers completed"
                )
            else:
                self.logger.debug(
                    f"Waiting for loggers to complete: {', '.join([f.function_name for f in active_functions])}"
                )

        for function in self._functions.values():
            if pending(function.status):
                self._handle_pending(function)
                if self._prev_statuses[function.function_name] != function.status:
                    self._log_status_change(function)
                    self._prev_statuses[function.function_name] = function.status

        if self._all_functions_completed():
            self.logger.info("All functions completed")
            raise StopMonitoring("All functions completed")

    ######################
    # Monitoring helpers #
    ######################
    def _handle_pending(self, function: FaaSrFunction) -> None:
        """Handle a pending function."""
        invocation_status = self._check_invocation_status(function)
        if invocation_status == InvocationStatus.INVOKED:
            self._reset_timer()
            function.set_status(FunctionStatus.INVOKED)
        elif invocation_status == InvocationStatus.NOT_INVOKED:
            self._reset_timer()
            function.set_status(FunctionStatus.NOT_INVOKED)

    def _log_status_change(self, function: FaaSrFunction) -> None:
        if failed(function.status):
            self.logger.info(f"Function {function.function_name} failed")
        elif not_invoked(function.status):
            self.logger.info(f"Function {function.function_name} not invoked")
        elif invoked(function.status):
            self.logger.info(f"Function {function.function_name} invoked")
        elif running(function.status):
            self.logger.info(f"Function {function.function_name} running")
        elif completed(function.status):
            self.logger.info(f"Function {function.function_name} completed")

    def _all_functions_completed(self) -> bool:
        """Check if all functions have completed."""
        return all(
            has_completed(function.status) for function in self._functions.values()
        )

    def _get_active_functions(self) -> list[FaaSrFunction]:
        """Get all functions that have logs started and are not complete.

        Only functions whose logger has actually started streaming count as
        "active" — a PENDING successor that was never invoked has no logger to
        wait for. Counting it as active would gate the failure cascade forever
        (it can never reach logs_complete or a final state on its own), hanging
        the monitor until timeout.
        """
        active = []
        for function in self._functions.values():
            if (
                function.logs_started
                and not function.logs_complete
                and not has_final_state(function.status)
            ):
                active.append(function)
        return active

    def _finish_monitoring(self) -> None:
        """Finish monitoring."""
        for function in self._functions.values():
            if not has_final_state(function.status) and self.shutdown_requested:
                function.set_status(FunctionStatus.SKIPPED)
                self.logger.info(
                    f"Function {function.function_name} skipped due to shutdown"
                )
            elif not has_final_state(function.status):
                function.set_status(FunctionStatus.TIMEOUT)
                self.logger.warning(f"Function {function.function_name} timed out")

        self._set_monitoring_complete()
        self.logger.info("Monitoring complete")

    def _cascade_failure(self) -> None:
        """Cascade a failure to all not completed functions."""
        for function in self._functions.values():
            if not has_final_state(function.status):
                function.set_status(FunctionStatus.SKIPPED)
                self.logger.info(
                    f"Skipping function {function.function_name} on failure"
                )

    ###################
    # Timeout helpers #
    ###################
    def _reset_timer(self) -> None:
        """Reset the monitoring timer."""
        self.last_change_time = time.time()
        self.seconds_since_last_change = 0.0

    def _increment_timer(self) -> None:
        """Increment the monitoring timer."""
        self.seconds_since_last_change = time.time() - self.last_change_time

    def _did_timeout(self) -> bool:
        """Check if the monitoring timer has timed out."""
        return self.seconds_since_last_change >= self.timeout

    #####################
    # Thread management #
    #####################
    def shutdown(self, timeout: float = None) -> bool:
        """Attempt to gracefully shutdown the monitoring thread."""
        self._set_shutdown_requested()

        if not self._monitoring_thread or not self._monitoring_thread.is_alive():
            return True

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.info("Requesting graceful shutdown of monitoring thread...")
            wait_timeout = timeout if timeout is not None else self._cleanup_timeout
            self._monitoring_thread.join(timeout=wait_timeout)

        if self._monitoring_thread.is_alive():
            self.logger.warning(
                f"Monitoring thread did not shutdown within {wait_timeout}s"
            )
            return False
        else:
            self.logger.info("Monitoring thread shutdown successfully")
            return True

    def force_shutdown(self) -> None:
        """Force shutdown of the monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Force shutting down monitoring and logs threads...")
            self._set_shutdown_requested()
            self._set_monitoring_complete()

    def _start(self):
        self._functions = self._build_functions(self._stream_logs)
        self._prev_statuses = self.get_function_statuses()

        self.logger.info(
            f"Workflow {self.workflow_name} triggered with InvocationID: {self._faasr_payload['InvocationID']}"
        )

        self._monitoring_thread = threading.Thread(
            target=self._start_monitoring,
            daemon=True,
        )
        self._monitoring_thread.start()

    ###################
    # Private helpers #
    ###################
    def _iter_ranks(self, function_name: str) -> Generator[str, None, None]:
        """Iterate over the ranks of a function."""
        rank_value = self.ranks.get(function_name, 1)
        if rank_value <= 1:
            yield function_name
        else:
            for rank in range(1, rank_value + 1):
                yield f"{function_name}({rank})"

    def _check_invocation_status(self, function: FaaSrFunction) -> InvocationStatus:
        """Check if a function was invoked across all of its invokers.

        - INVOKED if any invoker invoked it.
        - PENDING if no invoker invoked it yet but at least one is still pending.
        - NOT_INVOKED only once every invoker has definitively resolved to not
          invoking it (otherwise a fan-in successor could be wrongly resolved
          off the first invoker before the others have run).
        """
        any_pending = False
        for rank in chain(
            *(
                self._iter_ranks(invoker)
                for invoker in self.reverse_adj_graph[
                    extract_function_name(function.function_name)
                ]
            )
        ):
            status = self._get_invocation_status(self._functions[rank], function)
            if status == InvocationStatus.INVOKED:
                return InvocationStatus.INVOKED
            elif status == InvocationStatus.PENDING:
                any_pending = True
        return InvocationStatus.PENDING if any_pending else InvocationStatus.NOT_INVOKED

    def _edge_is_unconditional(
        self,
        invoker: FaaSrFunction,
        function: FaaSrFunction,
    ) -> bool:
        """Whether invoker→function is an unconditional InvokeNext edge.

        An edge is unconditional when the target appears as a plain string entry
        in the invoker's InvokeNext (e.g. "reduce" or "reduce(5)") rather than
        only inside a conditional ``{"True"/"False": [...]}`` dict. A completed
        invoker is guaranteed to fire its unconditional successors, so we can
        treat them as INVOKED without relying on fragile invoker-log parsing
        (which misses ranked fan-outs).
        """
        invoker_name = extract_function_name(invoker.function_name)
        target = extract_function_name(function.function_name)
        invoke_next = (
            self._faasr_payload["ActionList"].get(invoker_name, {}).get("InvokeNext")
        )
        if invoke_next is None:
            return False
        if isinstance(invoke_next, str):
            invoke_next = [invoke_next]
        for entry in invoke_next:
            # Plain string entries are unconditional; dicts are conditional branches.
            if isinstance(entry, str) and extract_function_name(entry) == target:
                return True
        return False

    def _get_invocation_status(
        self,
        invoker: FaaSrFunction,
        function: FaaSrFunction,
    ) -> Literal[InvocationStatus.PENDING, InvocationStatus.INVOKED] | None:
        """Get the invocation status of a function from a given invoker."""
        # A completed invoker WILL fire its unconditional successors. Trust the
        # static graph here instead of parsing the invoker's logs — log parsing
        # misses ranked fan-outs (e.g. shuffle → reduce(5)), which would wrongly
        # mark the successor NOT_INVOKED and let the monitor finish before it runs.
        if completed(invoker.status) and self._edge_is_unconditional(invoker, function):
            return InvocationStatus.INVOKED
        if invoker.invocations is not None:
            invocations = invoker.invocations
            if extract_function_name(function.function_name) in invocations:
                return InvocationStatus.INVOKED
            else:
                return InvocationStatus.NOT_INVOKED
        else:
            return InvocationStatus.PENDING
