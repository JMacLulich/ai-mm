"""Client adapter for the host-level Rust ``llm-router`` service."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .base import Provider, ProviderError, ProviderResponse

DEFAULT_LLM_ROUTER_BASE_URL = "http://127.0.0.1:4000/v1"
DEFAULT_LLM_ROUTER_TIMEOUT_SECONDS = 600.0

# llm-router owns the canonical four-level effort vocabulary. Legacy CLI spellings
# intentionally normalize into those semantic buckets; they are not model controls.
_EFFORT_MAP = {
    "none": "fast",
    "minimal": "fast",
    "low": "fast",
    "fast": "fast",
    "medium": "standard",
    "standard": "standard",
    "high": "careful",
    "careful": "careful",
    "xhigh": "max",
    "max": "max",
}


def _api_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        raise ProviderError("LLM_ROUTER_BASE_URL must not be empty")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProviderError("LLM_ROUTER_BASE_URL must be an http(s) URL")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _health_url(api_base_url: str) -> str:
    parsed = urlsplit(api_base_url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


class LLMRouterProvider(Provider):
    """The sole runtime LLM provider used by ``ai-mm``.

    The caller supplies a semantic route such as ``stage:review`` or
    ``profile:kimi``. Provider/model selection, retries, fallbacks, request quirks,
    and cost calculation remain inside the Rust router.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv("LLM_ROUTER_API_KEY") or "not-needed"
        self.base_url = _api_base_url(
            base_url or os.getenv("LLM_ROUTER_BASE_URL") or DEFAULT_LLM_ROUTER_BASE_URL
        )
        configured_timeout = timeout_seconds or os.getenv("LLM_ROUTER_TIMEOUT_SECONDS")
        try:
            self.timeout_seconds = float(
                configured_timeout or DEFAULT_LLM_ROUTER_TIMEOUT_SECONDS
            )
        except (TypeError, ValueError) as exc:
            raise ProviderError("LLM_ROUTER_TIMEOUT_SECONDS must be numeric") from exc
        if self.timeout_seconds <= 0:
            raise ProviderError("LLM_ROUTER_TIMEOUT_SECONDS must be greater than zero")

    def _sync_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ProviderError("openai package not installed. Run: pip install openai") from exc
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    def _async_client(self):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ProviderError("openai package not installed. Run: pip install openai") from exc
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    @staticmethod
    def _request_params(
        prompt: str,
        model: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if not (model.startswith("stage:") or model.startswith("profile:")):
            raise ProviderError(
                "llm-router requests require a semantic 'stage:' or 'profile:' intent"
            )
        if not 0.0 <= temperature <= 2.0:
            raise ProviderError("temperature must be between 0 and 2")

        metadata = dict(kwargs.pop("metadata", {}) or {})
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            try:
                metadata["effort"] = _EFFORT_MAP[str(reasoning_effort).lower()]
            except KeyError as exc:
                raise ProviderError(f"Unsupported router effort: {reasoning_effort}") from exc

        allowed = {"response_format", "tools", "tool_choice"}
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise ProviderError(
                "Unsupported llm-router request option(s): " + ", ".join(unknown)
            )

        params: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        for key in allowed:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        if metadata:
            params["extra_body"] = {"metadata": metadata}
        return params

    @staticmethod
    def _response(completion: Any, requested_route: str) -> ProviderResponse:
        choices = getattr(completion, "choices", None) or []
        content = choices[0].message.content if choices else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("llm-router returned empty response content")

        usage = getattr(completion, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        router = _as_dict(getattr(completion, "router", None))
        if not router:
            router = _as_dict(getattr(completion, "model_extra", None)).get("router", {})
            router = _as_dict(router)
        if not router:
            raise ProviderError(
                "Response lacks llm-router provenance; refusing an unverified routing result"
            )

        required_provenance = {
            "profile",
            "stage",
            "served_by_model_key",
            "served_by_model_id",
            "provider",
            "cost_usd",
            "fallback_outcome",
        }
        missing = sorted(key for key in required_provenance if router.get(key) is None)
        if missing:
            raise ProviderError(
                "llm-router provenance is incomplete; missing: " + ", ".join(missing)
            )

        served_model = str(router["served_by_model_id"])
        try:
            cost = Decimal(str(router["cost_usd"]))
        except Exception as exc:
            raise ProviderError("llm-router returned invalid cost provenance") from exc
        if not cost.is_finite() or cost < 0:
            raise ProviderError("llm-router returned invalid cost provenance")

        metadata = dict(router)
        metadata.update(
            {
                "requested_route": requested_route,
                "served_model": served_model,
                "router_verified": True,
            }
        )
        return ProviderResponse(
            text=content,
            model=served_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            cached=False,
            metadata=metadata,
        )

    def complete(
        self,
        prompt: str,
        model: str = "stage:review",
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        params = self._request_params(
            prompt, model, system_prompt, temperature, max_tokens, dict(kwargs)
        )
        try:
            completion = self._sync_client().chat.completions.create(**params)
            return self._response(completion, model)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"llm-router request failed: {exc}") from exc

    async def complete_async(
        self,
        prompt: str,
        model: str = "stage:review",
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        params = self._request_params(
            prompt, model, system_prompt, temperature, max_tokens, dict(kwargs)
        )
        try:
            completion = await self._async_client().chat.completions.create(**params)
            return self._response(completion, model)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"llm-router request failed: {exc}") from exc

    def get_model_info(self, model: str) -> dict[str, Any]:
        return {
            "provider": "llm_router",
            "route": model,
            "pricing": {},
            "routing_owner": "llm-router",
        }

    def health_check(self) -> dict[str, Any]:
        """Return the router's health payload without making an LLM call."""
        import json

        request = Request(_health_url(self.base_url), method="GET")
        try:
            with urlopen(request, timeout=min(self.timeout_seconds, 10.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderError(f"llm-router health check failed: {exc}") from exc
        if not isinstance(payload, dict) or not payload.get("any_available"):
            raise ProviderError("llm-router has no available route attempts")
        return payload
