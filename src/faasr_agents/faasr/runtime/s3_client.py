import queue
from typing import Any, Callable, ParamSpec, TypeVar

import boto3
from botocore.exceptions import ClientError

P = ParamSpec("P")
R = TypeVar("R")


class S3ClientInitializationError(Exception):
    """Exception raised for S3 client initialization errors"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"Error initializing S3 client: {self.message}"


class S3ClientError(Exception):
    """Exception raised for S3 client errors"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"S3 client error: {self.message}"


class FaaSrS3Client:
    """
    A client for interacting with FaaSr S3 datastores.

    This class is responsible for:
    - Initializing the S3 client
    - Checking if objects exist in S3
    - Getting objects from S3

    Args:
        workflow_data: The FaaSr workflow data.
        access_key: The FaaSr S3 access key.
        secret_key: The FaaSr S3 secret key.

    Raises:
        `S3ClientInitializationError`: If the S3 client initialization fails.
    """

    default_queue_size = 10
    default_timeout = 20

    def __init__(
        self,
        *,
        workflow_data: dict[str, Any],
        access_key: str,
        secret_key: str,
    ):
        try:
            default_datastore = workflow_data.get("DefaultDataStore", "S3")
            datastore_config = workflow_data["DataStores"][default_datastore]

            if datastore_config.get("Endpoint"):
                self._client = boto3.client(
                    "s3",
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=datastore_config["Region"],
                    endpoint_url=datastore_config["Endpoint"],
                )
            else:
                self._client = boto3.client(
                    "s3",
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=datastore_config["Region"],
                )

            self._bucket_name = datastore_config["Bucket"]

        except ClientError as e:
            raise S3ClientInitializationError(f"boto3 client error: {e}") from e
        except KeyError as e:
            raise S3ClientInitializationError(f"Key error: {e}") from e
        except Exception as e:
            raise S3ClientInitializationError(f"Unhandled error: {e}") from e

        self._queue = queue.Queue(maxsize=self.default_queue_size)
        self._timeout = self.default_timeout

        # Initialize the queue with a token for each allowed concurrent request
        for _ in range(self.default_queue_size):
            self._queue.put(object())

    def object_exists(self, key: str) -> bool:
        """
        Check if the object exists in S3.

        Args:
            key: The key of the object to check.

        Returns:
            True if the object exists, False otherwise.

        Raises:
            S3ClientError: If an error occurs.
        """
        return self._call(self._object_exists, key)

    def download_object(self, key: str, local_path: str) -> None:
        """
        Download an object from S3 to a local file (binary-safe).

        Args:
            key: The S3 key of the object to download.
            local_path: Local filesystem path to write the file to.

        Raises:
            S3ClientError: If the download fails.
        """
        return self._call(self._download_object, key, local_path)

    def list_objects(self, prefix: str) -> list[str]:
        """
        List the keys of all objects in S3 under a given prefix.

        Args:
            prefix: The key prefix to list under.

        Returns:
            The full keys of every object whose key starts with prefix.

        Raises:
            S3ClientError: If an error occurs.
        """
        return self._call(self._list_objects, prefix)

    def get_object(self, key: str, encoding: str = "utf-8") -> str:
        """
        Get the object from S3.

        Args:
            key: The key of the object to get.
            encoding: The encoding to use for the object.

        Returns:
            The object content.

        Raises:
            S3ClientError: If the object does not exist or an error occurs.
        """
        return self._call(self._get_object, key, encoding)

    def _object_exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket_name, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                raise S3ClientError(f"Error checking object existence: {e}") from e
        return True

    def _download_object(self, key: str, local_path: str) -> None:
        try:
            self._client.download_file(self._bucket_name, key, local_path)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise S3ClientError(f"Object does not exist: {key}") from e
            raise S3ClientError(f"boto3 client error downloading object: {e}") from e
        except Exception as e:
            raise S3ClientError(f"Unhandled error downloading object: {e}") from e

    def _list_objects(self, prefix: str) -> list[str]:
        try:
            keys: list[str] = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys
        except ClientError as e:
            raise S3ClientError(f"boto3 client error listing objects: {e}") from e
        except Exception as e:
            raise S3ClientError(f"Unhandled error listing objects: {e}") from e

    def _get_object(self, key: str, encoding: str = "utf-8") -> str:
        try:
            return (
                self._client.get_object(Bucket=self._bucket_name, Key=key)["Body"]
                .read()
                .decode(encoding)
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise S3ClientError(f"Object does not exist: {e}") from e
            raise S3ClientError(f"boto3 client error getting object: {e}") from e
        except Exception as e:
            raise S3ClientError(f"Unhandled error getting object: {e}") from e

    def _call(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Call the underlying client method and return the result. This pulls a token from
        the queue, blocking until one is available.

        Args:
            func: The function to call.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
        """
        token = self._queue.get(timeout=self._timeout)
        try:
            return func(*args, **kwargs)
        finally:
            self._queue.put(token)
