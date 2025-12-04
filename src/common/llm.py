"""Centralised LLM client utilities used across the project."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

load_dotenv()


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completion APIs.

    When the environment variable ``CASESENTINEL_MOCK_LLM`` is set, the client
    falls back to a deterministic mock response that simply truncates the user
    prompt. 这使得在离线环境中进行单元测试与开发成为可能。
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self.provider = self._resolve_provider()
        self.model = self._resolve_model(model)
        self._mock_mode = bool(os.getenv("CASESENTINEL_MOCK_LLM"))
        self._client = None

        if self._mock_mode:
            LOGGER.info("CASESENTINEL_MOCK_LLM set; using deterministic mock client")
            return

        try:
            self._client = self._initialise_client()
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.warning(
                "%s client initialization failed (%s); falling back to mock responses. "
                "Set CASESENTINEL_MOCK_LLM=1 或正确配置相关 API 凭据。",
                self.provider,
                exc,
            )
            self._mock_mode = True

    def _resolve_provider(self) -> str:
        explicit_provider = os.getenv("CASESENTINEL_LLM_PROVIDER")
        if explicit_provider:
            return explicit_provider.lower()

        if os.getenv("QWEN_API_KEY"):
            return "qwen"

        return "openai"

    def _resolve_model(self, explicit_model: Optional[str]) -> str:
        if explicit_model:
            return explicit_model

        env_override = os.getenv("CASESENTINEL_LLM_MODEL")
        if env_override:
            return env_override

        if self.provider == "qwen":
            return os.getenv("QWEN_MODEL_NAME", "qwen-plus")

        return os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    def _initialise_client(self):
        from openai import OpenAI  # type: ignore

        if self.provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY")
            if not api_key:
                raise RuntimeError("QWEN_API_KEY is required for provider 'qwen'")

            base_url = os.getenv(
                "QWEN_API_BASE",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            LOGGER.info("Initialising Qwen client via OpenAI-compatible endpoint at %s", base_url)
            return OpenAI(api_key=api_key, base_url=base_url)

        if self.provider == "openai":
            LOGGER.info("Initialising OpenAI client")
            return OpenAI()

        raise RuntimeError(f"Unsupported LLM provider: {self.provider}")

    @staticmethod
    def _truncate(text: str, limit: Optional[int]) -> str:
        if not limit or limit <= 0:
            return text
        return text[:limit]

    def _switch_to_mock(self, reason: Exception) -> None:
        if not self._mock_mode:
            LOGGER.warning("LLM request failed (%s); falling back to deterministic mock responses.", reason)
            self._mock_mode = True

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        if self._mock_mode:
            LOGGER.debug("Mock LLM mode enabled, returning heuristic narrative.")
            return self._truncate(user_prompt, kwargs.get("mock_truncate", 800))

        try:
            response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=kwargs.get("temperature", 0.4),
                max_tokens=kwargs.get("max_tokens"),
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - safeguard for offline mode
            self._switch_to_mock(exc)
            return self._truncate(user_prompt, kwargs.get("mock_truncate", 800))

    def generate_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Accept fully formatted chat messages (role/content)."""

        if self._mock_mode:
            combined = "\n".join(m["content"] for m in messages if m.get("role") == "user")
            return self._truncate(combined, kwargs.get("mock_truncate", 800))

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.4),
            "max_tokens": kwargs.get("max_tokens"),
        }
        if "response_format" in kwargs and kwargs["response_format"] is not None:
            request_kwargs["response_format"] = kwargs["response_format"]

        try:
            response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                **request_kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - safeguard for offline mode
            self._switch_to_mock(exc)
            combined = "\n".join(m["content"] for m in messages if m.get("role") == "user")
            return self._truncate(combined, kwargs.get("mock_truncate", 800))


__all__ = ["LLMClient"]
