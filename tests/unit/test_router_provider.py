"""Tests for the llm-router OpenAI-compatible adapter."""

import json
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from claude_mm.providers.base import ProviderError
from claude_mm.providers.router import LLMRouterProvider, _health_url


def _completion(router=None, content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5),
        model="served-id",
        router=router,
        model_extra={},
    )


def _provenance():
    return {
        "profile": "commercial_compliant",
        "stage": "review",
        "served_by_model_key": "router-key",
        "served_by_model_id": "served-id",
        "provider": "deepseek",
        "cost_usd": 0.012,
        "cost_source": "computed_estimate",
        "fallback_outcome": "served",
    }


def test_base_url_is_normalized_to_v1() -> None:
    provider = LLMRouterProvider(base_url="http://127.0.0.1:4000")
    assert provider.base_url == "http://127.0.0.1:4000/v1"


def test_health_url_uses_router_origin_for_custom_api_path() -> None:
    provider = LLMRouterProvider(base_url="http://127.0.0.1:4000/custom/v1/")
    assert provider.base_url == "http://127.0.0.1:4000/custom/v1"
    assert _health_url(provider.base_url) == "http://127.0.0.1:4000/health"


def test_base_url_rejects_non_http_scheme() -> None:
    with pytest.raises(ProviderError, match="http"):
        LLMRouterProvider(base_url="file:///tmp/router")


def test_request_maps_generic_effort_to_router_metadata() -> None:
    params = LLMRouterProvider._request_params(
        "prompt",
        "profile:commercial_compliant",
        "system",
        0.0,
        100,
        {"reasoning_effort": "xhigh", "metadata": {"operation": "review"}},
    )
    assert params["model"] == "profile:commercial_compliant"
    assert params["extra_body"]["metadata"] == {
        "operation": "review",
        "effort": "max",
    }


def test_request_rejects_raw_model_and_provider_specific_options() -> None:
    with pytest.raises(ProviderError, match="semantic"):
        LLMRouterProvider._request_params("p", "model-id", None, 0, None, {})
    with pytest.raises(ProviderError, match="Unsupported"):
        LLMRouterProvider._request_params(
            "p", "stage:review", None, 0, None, {"reasoning_content": True}
        )
    with pytest.raises(ProviderError, match="temperature"):
        LLMRouterProvider._request_params("p", "stage:review", None, 2.1, None, {})


def test_response_requires_router_provenance() -> None:
    with pytest.raises(ProviderError, match="provenance"):
        LLMRouterProvider._response(_completion(router=None), "stage:review")
    incomplete = _provenance()
    incomplete.pop("served_by_model_key")
    with pytest.raises(ProviderError, match="served_by_model_key"):
        LLMRouterProvider._response(_completion(router=incomplete), "stage:review")


def test_response_uses_router_cost_and_served_model() -> None:
    response = LLMRouterProvider._response(
        _completion(router=_provenance()), "profile:commercial_compliant"
    )
    assert response.model == "served-id"
    assert response.input_tokens == 3
    assert response.output_tokens == 5
    assert response.cost == Decimal("0.012")
    assert response.metadata["router_verified"] is True
    assert response.metadata["requested_route"] == "profile:commercial_compliant"


def test_complete_calls_router_compatibility_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _completion(router=_provenance())

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    provider = LLMRouterProvider(base_url="http://127.0.0.1:4000/v1")
    monkeypatch.setattr(provider, "_sync_client", lambda: client)
    result = provider.complete("patch", "stage:review", reasoning_effort="careful")
    assert result.text == "ok"
    assert captured["model"] == "stage:review"
    assert captured["extra_body"]["metadata"]["effort"] == "careful"


def test_real_http_boundary_exercises_health_completion_and_provenance() -> None:
    observed: dict[str, object] = {}

    class RouterHandler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            observed["health_path"] = self.path
            self._send_json(
                {
                    "any_available": True,
                    "entries": [{"model_key": "internal-flash", "status": "up"}],
                }
            )

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            observed["completion_path"] = self.path
            observed["request"] = json.loads(self.rfile.read(length))
            self._send_json(
                {
                    "id": "chatcmpl-integration",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "internal-upstream-id",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "reviewed"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "total_tokens": 9,
                    },
                    "router": {
                        "profile": "coding",
                        "stage": "review",
                        "served_by_model_key": "internal-flash",
                        "served_by_model_id": "internal-upstream-id",
                        "provider": "deepseek",
                        "cost_usd": 0.0004,
                        "fallback_outcome": "served",
                    },
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        provider = LLMRouterProvider(base_url=f"http://{host}:{port}/custom/v1")
        assert provider.health_check()["any_available"] is True
        response = provider.complete(
            "patch", "stage:review", reasoning_effort="minimal", max_tokens=50
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert observed["health_path"] == "/health"
    assert observed["completion_path"] == "/custom/v1/chat/completions"
    assert observed["request"]["model"] == "stage:review"
    assert observed["request"]["metadata"]["effort"] == "fast"
    assert response.text == "reviewed"
    assert response.model == "internal-upstream-id"
    assert response.cost == Decimal("0.0004")
    assert response.metadata["router_verified"] is True
