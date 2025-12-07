"""Centralised LLM client utilities with MindSpore/Ascend support."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

load_dotenv()


def _read_bool_env(*names: str) -> bool:
    """Return True if any of the provided environment variables is truthy."""

    for name in names:
        if os.getenv(name):
            return True
    return False


def _truncate(text: str, limit: Optional[int]) -> str:
    if not limit or limit <= 0:
        return text
    return text[:limit]


class MindSporeChatBackend:
    """MindSpore/Ascend-backed chat generation.

    Uses ``mindformers.pipeline`` for text generation. Errors during import or
    model loading are propagated to allow the caller to fallback to mock mode.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device_target: str = "Ascend",
        precision: str = "float16",
        max_length: int = 2048,
        graph_mode: bool = True,
    ) -> None:
        import mindspore as ms  # type: ignore
        from mindformers import pipeline  # type: ignore

        mode = ms.GRAPH_MODE if graph_mode else ms.PYNATIVE_MODE
        ms.set_context(mode=mode, device_target=device_target)
        LOGGER.info(
            "Initialising MindSpore backend (device=%s, precision=%s, graph_mode=%s) with model %s",
            device_target,
            precision,
            graph_mode,
            model_path,
        )
        self._generator = pipeline(task="text_generation", model=model_path)
        self.model_name = model_path
        self.max_length = max_length
        self.precision = precision

    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate_chat(messages, **kwargs)

    def generate_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        prompt = self._format_messages(messages)
        temperature = kwargs.get("temperature", 0.4)
        max_tokens = kwargs.get("max_tokens")
        max_length = max_tokens + len(prompt.split()) if max_tokens else self.max_length
        result = self._generator(
            prompt,
            max_length=max_length,
            do_sample=temperature > 0.05,
            top_p=kwargs.get("top_p", 0.9),
            temperature=temperature,
        )
        text = self._extract_text(result)
        return text.strip()

    @staticmethod
    def _extract_text(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, dict) and "text" in item:
                return str(item.get("text", ""))
            if isinstance(item, str):
                return item
        if isinstance(result, dict) and "text" in result:
            return str(result.get("text", ""))
        return ""


class OpenAICompatBackend:
    """OpenAI-compatible backend retained for optional use."""

    def __init__(self, provider: str, model: str) -> None:
        from openai import OpenAI  # type: ignore

        self.provider = provider
        self.model = model

        if provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY")
            if not api_key:
                raise RuntimeError("QWEN_API_KEY is required for provider 'qwen'")
            base_url = os.getenv(
                "QWEN_API_BASE",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            LOGGER.info("Initialising Qwen client via OpenAI-compatible endpoint at %s", base_url)
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            LOGGER.info("Initialising OpenAI client")
            self._client = OpenAI()

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
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

    def generate_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.4),
            "max_tokens": kwargs.get("max_tokens"),
        }
        if "response_format" in kwargs and kwargs["response_format"] is not None:
            request_kwargs["response_format"] = kwargs["response_format"]

        response = self._client.chat.completions.create(  # type: ignore[attr-defined]
            **request_kwargs,
        )
        return response.choices[0].message.content or ""


class LLMClient:
    """Unified LLM client with MindSpore/Ascend priority.

    When ``CASESENTINEL_MOCK_LLM`` or ``CRIMEMENTOR_MOCK_LLM`` is set, the
    client falls back to deterministic mock responses that truncate prompts.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self.provider = self._resolve_provider()
        self.model = self._resolve_model(model)
        self._mock_mode = _read_bool_env("CASESENTINEL_MOCK_LLM", "CRIMEMENTOR_MOCK_LLM")
        self._backend: Optional[Any] = None

        if self._mock_mode:
            LOGGER.info("Mock LLM mode enabled; deterministic responses will be used")
            return

        try:
            self._backend = self._initialise_backend()
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.warning(
                "%s backend initialization failed (%s); falling back to mock responses. "
                "Set CASESENTINEL_MOCK_LLM=1 或正确配置相关模型与设备。",
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

        return "mindspore"

    def _resolve_model(self, explicit_model: Optional[str]) -> str:
        if explicit_model:
            return explicit_model

        env_override = os.getenv("CASESENTINEL_LLM_MODEL")
        if env_override:
            return env_override

        if self.provider == "qwen":
            return os.getenv("QWEN_MODEL_NAME", "qwen-plus")
        if self.provider == "openai":
            return os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

        return os.getenv("CASESENTINEL_MINDSPORE_MODEL", "glm3-6b")

    def _initialise_backend(self):
        if self.provider in {"openai", "qwen"}:
            return OpenAICompatBackend(self.provider, self.model)

        if self.provider == "mindspore":
            device = os.getenv("CASESENTINEL_DEVICE", "Ascend")
            precision = os.getenv("CASESENTINEL_MINDSPORE_PRECISION", "float16")
            max_length = int(os.getenv("CASESENTINEL_MINDSPORE_MAX_LENGTH", "2048"))
            graph_mode = os.getenv("CASESENTINEL_MINDSPORE_GRAPH_MODE", "1") not in {"0", "false", "False"}
            return MindSporeChatBackend(
                self.model,
                device_target=device,
                precision=precision,
                max_length=max_length,
                graph_mode=graph_mode,
            )

        raise RuntimeError(f"Unsupported LLM provider: {self.provider}")

    def _switch_to_mock(self, reason: Exception) -> None:
        if not self._mock_mode:
            LOGGER.warning("LLM request failed (%s); falling back to deterministic mock responses.", reason)
            self._mock_mode = True

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        if self._mock_mode or not self._backend:
            LOGGER.debug("Mock LLM mode enabled, returning heuristic narrative.")
            return _truncate(user_prompt, kwargs.get("mock_truncate", 800))

        try:
            return self._backend.generate(system_prompt, user_prompt, **kwargs)
        except Exception as exc:  # pragma: no cover - safeguard for offline mode
            self._switch_to_mock(exc)
            return _truncate(user_prompt, kwargs.get("mock_truncate", 800))

    def generate_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Accept fully formatted chat messages (role/content)."""

        if self._mock_mode or not self._backend:
            combined = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            return _truncate(combined, kwargs.get("mock_truncate", 800))

        try:
            return self._backend.generate_chat(messages, **kwargs)
        except Exception as exc:  # pragma: no cover - safeguard for offline mode
            self._switch_to_mock(exc)
            combined = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            return _truncate(combined, kwargs.get("mock_truncate", 800))


__all__ = ["LLMClient"]
