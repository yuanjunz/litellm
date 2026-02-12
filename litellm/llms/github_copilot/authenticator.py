import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    GetAccessTokenError,
    GetAPIKeyError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
)

# Constants
GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_API_KEY_URL = "https://api.github.com/copilot_internal/v2/token"

# Background refresh constants
EARLY_REFRESH_SECONDS = 60  # refresh 60s before expiry


class Authenticator:
    """GitHub Copilot authenticator with singleton pattern, in-memory caching,
    and background token refresh."""

    _instance: Optional["Authenticator"] = None
    _instance_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "Authenticator":
        """Return the shared singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton. For testing only."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance._cancel_timer()
            cls._instance = None

    def __init__(self) -> None:
        """Initialize the GitHub Copilot authenticator with configurable token paths."""
        # Token storage paths
        self.token_dir = os.getenv(
            "GITHUB_COPILOT_TOKEN_DIR",
            os.path.expanduser("~/.config/litellm/github_copilot"),
        )
        self.access_token_file = os.path.join(
            self.token_dir,
            os.getenv("GITHUB_COPILOT_ACCESS_TOKEN_FILE", "access-token"),
        )
        self.api_key_file = os.path.join(
            self.token_dir, os.getenv("GITHUB_COPILOT_API_KEY_FILE", "api-key.json")
        )

        # In-memory caches
        self._cached_api_key_info: Optional[Dict[str, Any]] = None
        self._cached_access_token: Optional[str] = None

        # Concurrency control
        self._refresh_lock: threading.Lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

        self._ensure_token_dir()
        self._load_cached_state()

    # ── Public API ──────────────────────────────────────────────────────

    def get_access_token(self) -> str:
        """
        Get the GitHub OAuth access token, using cache when available.

        Returns:
            str: The GitHub access token.

        Raises:
            GetAccessTokenError: If unable to obtain an access token after retries.
        """
        if self._cached_access_token:
            return self._cached_access_token

        try:
            with open(self.access_token_file, "r") as f:
                access_token = f.read().strip()
                if access_token:
                    self._cached_access_token = access_token
                    return access_token
        except IOError:
            verbose_logger.warning(
                "No existing access token found or error reading file"
            )

        for attempt in range(3):
            verbose_logger.debug(f"Access token acquisition attempt {attempt + 1}/3")
            try:
                access_token = self._login()
                self._cached_access_token = access_token
                try:
                    with open(self.access_token_file, "w") as f:
                        f.write(access_token)
                except IOError:
                    verbose_logger.error("Error saving access token to file")
                return access_token
            except (GetDeviceCodeError, GetAccessTokenError, RefreshAPIKeyError) as e:
                verbose_logger.warning(f"Failed attempt {attempt + 1}: {str(e)}")
                continue

        raise GetAccessTokenError(
            message="Failed to get access token after 3 attempts",
            status_code=401,
        )

    def get_api_key(self) -> str:
        """
        Get the API key, serving from cache when valid.

        Fast path (cache hit): lock-free dict read + timestamp comparison.
        Slow path (cache miss/expired): acquires lock, refreshes, updates cache.

        Returns:
            str: The GitHub Copilot API key.

        Raises:
            GetAPIKeyError: If unable to obtain an API key.
        """
        # Fast path: return cached token if still valid (no lock, no I/O)
        if self._is_cached_token_valid():
            return self._cached_api_key_info["token"]  # type: ignore[index]

        # Slow path: acquire lock to prevent thundering herd
        with self._refresh_lock:
            # Double-check after acquiring lock (another thread may have refreshed)
            if self._is_cached_token_valid():
                return self._cached_api_key_info["token"]  # type: ignore[index]

            # Try loading from file (another process may have refreshed)
            self._load_api_key_from_file()
            if self._is_cached_token_valid():
                self._schedule_background_refresh()
                return self._cached_api_key_info["token"]  # type: ignore[index]

            # Perform synchronous HTTP refresh
            try:
                api_key_info = self._refresh_api_key()
                self._cached_api_key_info = api_key_info
                self._save_api_key_to_file(api_key_info)
                self._schedule_background_refresh()
                token = api_key_info.get("token")
                if token:
                    return token
                raise GetAPIKeyError(
                    message="API key response missing token",
                    status_code=401,
                )
            except RefreshAPIKeyError as e:
                raise GetAPIKeyError(
                    message=f"Failed to refresh API key: {str(e)}",
                    status_code=401,
                )

    def get_api_base(self) -> Optional[str]:
        """
        Get the API endpoint, preferring cached data.

        Returns:
            Optional[str]: The GitHub Copilot API endpoint, or None if not found.
        """
        if self._cached_api_key_info is not None:
            endpoints = self._cached_api_key_info.get("endpoints", {})
            api_endpoint = endpoints.get("api")
            if api_endpoint:
                return api_endpoint

        try:
            with open(self.api_key_file, "r") as f:
                api_key_info = json.load(f)
                endpoints = api_key_info.get("endpoints", {})
                return endpoints.get("api")
        except (IOError, json.JSONDecodeError, KeyError) as e:
            verbose_logger.warning(f"Error reading API endpoint from file: {str(e)}")
            return None

    # ── Background refresh ──────────────────────────────────────────────

    def _schedule_background_refresh(self) -> None:
        """Schedule a daemon timer to refresh the token before it expires."""
        if self._cached_api_key_info is None:
            return

        expires_at = self._cached_api_key_info.get("expires_at", 0)
        now = datetime.now().timestamp()
        delay = max(1.0, expires_at - now - EARLY_REFRESH_SECONDS)

        self._cancel_timer()
        self._timer = threading.Timer(delay, self._do_background_refresh)
        self._timer.daemon = True
        self._timer.start()

        verbose_logger.debug(
            f"GitHub Copilot: background refresh scheduled in {delay:.0f}s "
            f"(token expires in {expires_at - now:.0f}s)"
        )

    def _do_background_refresh(self) -> None:
        """Perform background token refresh with retries."""
        for retry in range(3):
            try:
                api_key_info = self._refresh_api_key()
                with self._refresh_lock:
                    self._cached_api_key_info = api_key_info
                self._save_api_key_to_file(api_key_info)
                verbose_logger.info(
                    "GitHub Copilot: background token refresh succeeded"
                )
                self._schedule_background_refresh()
                return
            except Exception as e:
                verbose_logger.warning(
                    f"GitHub Copilot: background refresh attempt {retry + 1}/3 failed: {e}"
                )
                if retry < 2:
                    time.sleep(10)

        verbose_logger.error(
            "GitHub Copilot: background refresh exhausted all retries. "
            "On-demand refresh will be attempted on next request."
        )

    def _cancel_timer(self) -> None:
        """Cancel the pending background refresh timer if any."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    # ── Token refresh (HTTP) ────────────────────────────────────────────

    def _refresh_api_key(self) -> Dict[str, Any]:
        """
        Refresh the API key using the access token, with exponential backoff.

        Returns:
            Dict[str, Any]: The API key information including token and expiration.

        Raises:
            RefreshAPIKeyError: If unable to refresh the API key.
        """
        access_token = self.get_access_token()
        headers = self._get_github_headers(access_token)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                sync_client = _get_httpx_client()
                response = sync_client.get(GITHUB_API_KEY_URL, headers=headers)
                response.raise_for_status()

                response_json = response.json()

                if "token" in response_json:
                    return response_json
                else:
                    verbose_logger.warning(
                        f"API key response missing token: {response_json}"
                    )
            except httpx.HTTPStatusError as e:
                verbose_logger.error(
                    f"HTTP error refreshing API key (attempt {attempt+1}/{max_retries}): {str(e)}"
                )
            except Exception as e:
                verbose_logger.error(f"Unexpected error refreshing API key: {str(e)}")

            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 5))  # 1s, 2s backoff

        raise RefreshAPIKeyError(
            message="Failed to refresh API key after maximum retries",
            status_code=401,
        )

    # ── Cache helpers ───────────────────────────────────────────────────

    def _is_cached_token_valid(self) -> bool:
        """Check if the in-memory cached API key is present and not expired."""
        if self._cached_api_key_info is None:
            return False
        token = self._cached_api_key_info.get("token")
        if not token:
            return False
        expires_at = self._cached_api_key_info.get("expires_at", 0)
        return expires_at > datetime.now().timestamp()

    def _load_cached_state(self) -> None:
        """Best-effort load of existing token files into memory at startup."""
        # Load access token
        try:
            with open(self.access_token_file, "r") as f:
                token = f.read().strip()
                if token:
                    self._cached_access_token = token
        except IOError:
            pass

        # Load API key info
        self._load_api_key_from_file()

        # If we have a valid cached token, start background refresh
        if self._is_cached_token_valid():
            self._schedule_background_refresh()

    def _load_api_key_from_file(self) -> None:
        """Load api-key.json into _cached_api_key_info (best-effort)."""
        try:
            with open(self.api_key_file, "r") as f:
                api_key_info = json.load(f)
                if isinstance(api_key_info, dict) and "token" in api_key_info:
                    self._cached_api_key_info = api_key_info
        except (IOError, json.JSONDecodeError):
            pass

    def _save_api_key_to_file(self, api_key_info: Dict[str, Any]) -> None:
        """Persist api_key_info to disk (best-effort, errors logged)."""
        try:
            with open(self.api_key_file, "w") as f:
                json.dump(api_key_info, f)
        except IOError as e:
            verbose_logger.error(f"Error saving API key to file: {str(e)}")

    # ── Infrastructure ──────────────────────────────────────────────────

    def _ensure_token_dir(self) -> None:
        """Ensure the token directory exists."""
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    def _get_github_headers(self, access_token: Optional[str] = None) -> Dict[str, str]:
        """
        Generate standard GitHub headers for API requests.

        Args:
            access_token: Optional access token to include in the headers.

        Returns:
            Dict[str, str]: Headers for GitHub API requests.
        """
        headers = {
            "accept": "application/json",
            "editor-version": "vscode/1.85.1",
            "editor-plugin-version": "copilot/1.155.0",
            "user-agent": "GithubCopilot/1.155.0",
            "accept-encoding": "gzip,deflate,br",
        }

        if access_token:
            headers["authorization"] = f"token {access_token}"

        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        return headers

    def _get_device_code(self) -> Dict[str, str]:
        """
        Get a device code for GitHub authentication.

        Returns:
            Dict[str, str]: Device code information.

        Raises:
            GetDeviceCodeError: If unable to get a device code.
        """
        try:
            sync_client = _get_httpx_client()
            resp = sync_client.post(
                GITHUB_DEVICE_CODE_URL,
                headers=self._get_github_headers(),
                json={"client_id": GITHUB_CLIENT_ID, "scope": "read:user"},
            )
            resp.raise_for_status()
            resp_json = resp.json()

            required_fields = ["device_code", "user_code", "verification_uri"]
            if not all(field in resp_json for field in required_fields):
                verbose_logger.error(f"Response missing required fields: {resp_json}")
                raise GetDeviceCodeError(
                    message="Response missing required fields",
                    status_code=400,
                )

            return resp_json
        except httpx.HTTPStatusError as e:
            verbose_logger.error(f"HTTP error getting device code: {str(e)}")
            raise GetDeviceCodeError(
                message=f"Failed to get device code: {str(e)}",
                status_code=400,
            )
        except json.JSONDecodeError as e:
            verbose_logger.error(f"Error decoding JSON response: {str(e)}")
            raise GetDeviceCodeError(
                message=f"Failed to decode device code response: {str(e)}",
                status_code=400,
            )
        except Exception as e:
            verbose_logger.error(f"Unexpected error getting device code: {str(e)}")
            raise GetDeviceCodeError(
                message=f"Failed to get device code: {str(e)}",
                status_code=400,
            )

    def _poll_for_access_token(self, device_code: str) -> str:
        """
        Poll for an access token after user authentication.

        Args:
            device_code: The device code to use for polling.

        Returns:
            str: The access token.

        Raises:
            GetAccessTokenError: If unable to get an access token.
        """
        sync_client = _get_httpx_client()
        max_attempts = 12  # 1 minute (12 * 5 seconds)

        for attempt in range(max_attempts):
            try:
                resp = sync_client.post(
                    GITHUB_ACCESS_TOKEN_URL,
                    headers=self._get_github_headers(),
                    json={
                        "client_id": GITHUB_CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                )
                resp.raise_for_status()
                resp_json = resp.json()

                if "access_token" in resp_json:
                    verbose_logger.info("Authentication successful!")
                    return resp_json["access_token"]
                elif (
                    "error" in resp_json
                    and resp_json.get("error") == "authorization_pending"
                ):
                    verbose_logger.debug(
                        f"Authorization pending (attempt {attempt+1}/{max_attempts})"
                    )
                else:
                    verbose_logger.warning(f"Unexpected response: {resp_json}")
            except httpx.HTTPStatusError as e:
                verbose_logger.error(f"HTTP error polling for access token: {str(e)}")
                raise GetAccessTokenError(
                    message=f"Failed to get access token: {str(e)}",
                    status_code=400,
                )
            except json.JSONDecodeError as e:
                verbose_logger.error(f"Error decoding JSON response: {str(e)}")
                raise GetAccessTokenError(
                    message=f"Failed to decode access token response: {str(e)}",
                    status_code=400,
                )
            except Exception as e:
                verbose_logger.error(
                    f"Unexpected error polling for access token: {str(e)}"
                )
                raise GetAccessTokenError(
                    message=f"Failed to get access token: {str(e)}",
                    status_code=400,
                )

            time.sleep(5)

        raise GetAccessTokenError(
            message="Timed out waiting for user to authorize the device",
            status_code=400,
        )

    def _login(self) -> str:
        """
        Login to GitHub Copilot using device code flow.

        Returns:
            str: The GitHub access token.

        Raises:
            GetDeviceCodeError: If unable to get a device code.
            GetAccessTokenError: If unable to get an access token.
        """
        device_code_info = self._get_device_code()

        device_code = device_code_info["device_code"]
        user_code = device_code_info["user_code"]
        verification_uri = device_code_info["verification_uri"]

        print(  # noqa: T201
            f"Please visit {verification_uri} and enter code {user_code} to authenticate.",

            # When this is running in docker, it may not be flushed immediately
            # so we force flush to ensure the user sees the message
            flush=True,
        )

        return self._poll_for_access_token(device_code)
