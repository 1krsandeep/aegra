"""The Open Responses router is mounted only when http.responses is true."""

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from aegra_api.config import responses_enabled
from aegra_api.main import _include_core_routers, create_app


def _paths(app: FastAPI) -> set[str]:
    found: set[str] = set()

    def walk(routes: list[object]) -> None:
        for route in routes:
            path = getattr(route, "path", None)
            if isinstance(route, APIRoute) and isinstance(path, str):
                found.add(path)
            nested = getattr(route, "original_router", None)
            if nested is not None:
                walk(list(nested.routes))
            elif hasattr(route, "routes"):
                walk(list(route.routes))

    walk(list(app.routes))
    return found


def test_responses_enabled_requires_boolean_true() -> None:
    """Absent, false, and non-boolean values leave the surface off."""
    assert responses_enabled(None) is False
    assert responses_enabled({}) is False
    assert responses_enabled({"responses": False}) is False
    assert responses_enabled({"responses": "true"}) is False
    assert responses_enabled({"responses": True}) is True


def test_core_routers_omit_responses_unless_asked() -> None:
    """The flag is what mounts /v1, not merely importing the router."""
    disabled = FastAPI()
    _include_core_routers(disabled)
    enabled = FastAPI()
    _include_core_routers(enabled, include_responses=True)

    assert "/v1/responses" not in _paths(disabled)
    assert "/v1/responses" in _paths(enabled)
    assert "/v1/conversations" in _paths(enabled)


def test_create_app_honors_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_app reads http.responses and does not mount the routes when it is off."""
    monkeypatch.setattr("aegra_api.main.load_http_config", lambda: {"responses": False})
    disabled = create_app()
    monkeypatch.setattr("aegra_api.main.load_http_config", lambda: {"responses": True})
    enabled = create_app()

    assert "/v1/responses" not in _paths(disabled)
    assert "/v1/responses" in _paths(enabled)
