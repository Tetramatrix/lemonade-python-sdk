"""
Reusable Lemonade model recovery helper.

This helper handles the common failure modes for configured-but-not-ready models:
- model not yet visible in the live list
- model needs a warm-up probe
- model is stuck and needs unload/reload before it becomes usable
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from .client import LemonadeClient
from .model_info import ModelInfo

logger = logging.getLogger(__name__)


class LemonadeModelRecovery:
    """Recover Lemonade models into a ready state for reuse across projects."""

    _RECOVERABLE_HINTS = (
        "not in list",
        "does not support embeddings",
        "--embeddings",
        "couldn't connect",
        "connection refused",
        "failure when receiving",
        "timeout",
        "model not found",
    )

    def __init__(
        self,
        base_url: str,
        client: Optional[LemonadeClient] = None,
        warmup_text: str = "test",
        allow_model_management: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.warmup_text = warmup_text
        self.allow_model_management = allow_model_management

    @property
    def client(self) -> Optional[LemonadeClient]:
        """Return a cached LemonadeClient or build one on demand."""
        if self._client is None:
            try:
                self._client = LemonadeClient(base_url=self.base_url)
            except Exception as e:
                logger.debug(f"[LemonadeModelRecovery] Failed to create client: {e}")
                self._client = None
        return self._client

    def ensure_model_ready(self, model_name: str, *, recover: bool = True) -> bool:
        """Compatibility wrapper for callers that only care about model availability."""
        return self.ensure_model_info_ready(model_name, recover=recover)

    def ensure_model_info_ready(self, model_name: str, *, recover: bool = True) -> bool:
        """
        Ensure the model appears in the live model info list.

        This is the right check for capability queries such as vision support.
        """
        if not model_name:
            return False

        if self._get_model_info(model_name) is not None:
            return True

        if not recover:
            return False

        if self._load_and_recheck_model_info(model_name):
            return True

        if self._model_management_supported is False:
            return False

        return self._unload_reload_and_recheck_model_info(model_name)

    def ensure_embedding_ready(self, model_name: str, *, recover: bool = True) -> bool:
        """
        Ensure the embedding model can actually serve an embedding request.

        This is stricter than model-info readiness because some Lemonade states
        expose a model in metadata but still fail when embeddings are requested.
        """
        if not model_name:
            return False

        if self._probe_embedding_with_retries(model_name):
            return True

        if not recover:
            return False

        if not self.allow_model_management:
            logger.debug(
                f"[LemonadeModelRecovery] Model management disabled; using probe-only mode for "
                f"'{model_name}' at {self.base_url}"
            )
            return False

        if self._load_and_reprobe_embedding(model_name):
            return True

        return self._unload_reload_and_reprobe_embedding(model_name)

    def _get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        client = self.client
        if client is None:
            return None

        try:
            return client.get_model_info(model_name)
        except Exception as e:
            logger.debug(f"[LemonadeModelRecovery] get_model_info failed: {e}")
            return None

    def _probe_embedding(self, model_name: str) -> bool:
        client = self.client
        if client is None:
            return False

        try:
            response = client.embeddings(
                input=self.warmup_text,
                model=model_name,
                encoding_format="float",
            )

            if response is None:
                logger.warning(
                    f"[LemonadeModelRecovery] Embedding probe returned None from "
                    f"{self.base_url}/api/v1/embeddings for model '{model_name}'"
                )
                return False

            if isinstance(response, dict) and "error" in response:
                error_text = str(response.get("error", "")).lower()
                logger.debug(f"[LemonadeModelRecovery] Embedding probe returned error: {error_text}")
                return False

            if isinstance(response, dict):
                if "data" in response or "embedding" in response:
                    return True
                logger.debug(
                    f"[LemonadeModelRecovery] Embedding probe returned unexpected dict from "
                    f"{self.base_url}/api/v1/embeddings for model '{model_name}': {response.keys()}"
                )
                return False

            if isinstance(response, list):
                return len(response) > 0

            logger.debug(
                f"[LemonadeModelRecovery] Embedding probe returned unexpected type from "
                f"{self.base_url}/api/v1/embeddings for model '{model_name}': {type(response)}"
            )
            return False
        except Exception as e:
            logger.debug(
                f"[LemonadeModelRecovery] Embedding probe failed for "
                f"{self.base_url}/api/v1/embeddings and model '{model_name}': {e}"
            )
            return False

    def _probe_embedding_with_retries(self, model_name: str, retries: int = 3) -> bool:
        """Retry a direct embedding probe a few times before escalating."""
        delay = 0.5
        for attempt in range(1, retries + 1):
            if self._probe_embedding(model_name):
                return True
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
        return False

    def _load_and_recheck_model_info(self, model_name: str) -> bool:
        client = self.client
        if client is None:
            return False
        if not self.allow_model_management:
            return False

        try:
            logger.info(f"[LemonadeModelRecovery] Loading model for info recovery: {model_name}")
            load_result = self._post_model_management("/api/v1/load", {"model": model_name})
            logger.info(f"[LemonadeModelRecovery] Load result: {load_result}")
        except Exception as e:
            logger.debug(f"[LemonadeModelRecovery] load_model failed: {e}")

        return self._get_model_info(model_name) is not None

    def _unload_reload_and_recheck_model_info(self, model_name: str) -> bool:
        client = self.client
        if client is None:
            return False
        if not self.allow_model_management:
            return False

        try:
            logger.warning(
                f"[LemonadeModelRecovery] Forcing unload/reload for info recovery: {model_name}"
            )
            self._unload_and_load(client, model_name)
        except Exception as e:
            logger.debug(f"[LemonadeModelRecovery] unload/reload info recovery failed: {e}")

        return self._get_model_info(model_name) is not None

    def _load_and_reprobe_embedding(self, model_name: str) -> bool:
        client = self.client
        if client is None:
            return False
        if not self.allow_model_management:
            return False

        try:
            logger.info(f"[LemonadeModelRecovery] Loading model for embedding recovery: {model_name}")
            load_result = self._post_model_management("/api/v1/load", {"model": model_name})
            logger.info(f"[LemonadeModelRecovery] Load result: {load_result}")
        except Exception as e:
            logger.debug(f"[LemonadeModelRecovery] load_model failed: {e}")

        return self._probe_embedding(model_name)

    def _unload_reload_and_reprobe_embedding(self, model_name: str) -> bool:
        client = self.client
        if client is None:
            return False
        if not self.allow_model_management:
            return False

        try:
            logger.warning(
                f"[LemonadeModelRecovery] Forcing unload/reload for embedding recovery: {model_name}"
            )
            self._unload_and_load(client, model_name)
        except Exception as e:
            logger.debug(f"[LemonadeModelRecovery] unload/reload embedding recovery failed: {e}")

        return self._probe_embedding(model_name)

    def _unload_and_load(self, client: LemonadeClient, model_name: str) -> None:
        """Best-effort unload/reload pair used by both recovery modes."""
        try:
            unload_result = self._post_model_management("/api/v1/unload", {})
            logger.info(f"[LemonadeModelRecovery] Unload result: {unload_result}")
        except Exception as e:
            logger.debug(f"[LemonadeModelRecovery] unload_model failed: {e}")

        try:
            load_result = self._post_model_management("/api/v1/load", {"model": model_name})
            logger.info(f"[LemonadeModelRecovery] Reload result: {load_result}")
        except Exception as e:
            logger.debug(f"[LemonadeModelRecovery] reload_model failed: {e}")

    def _post_model_management(self, endpoint: str, payload: dict) -> dict:
        """Call Lemonade model management endpoints directly on the shared base URL."""
        client = self.client
        if client is None:
            return {"error": "Lemonade client not available"}

        url = f"{self.base_url}{endpoint}"
        try:
            response = client.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            if response.content:
                try:
                    return response.json()
                except Exception:
                    return {"status": response.text}
            return {}
        except requests.RequestException as e:
            logger.info(f"[LemonadeModelRecovery] Management request failed for {url}: {e}")
            return {"error": str(e)}

    @classmethod
    def is_recoverable_error(cls, error_text: str) -> bool:
        """Return True when an error string looks like a recoverable Lemonade state."""
        lowered = (error_text or "").lower()
        return any(hint in lowered for hint in cls._RECOVERABLE_HINTS)
