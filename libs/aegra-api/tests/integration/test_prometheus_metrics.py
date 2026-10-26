"""Integration tests for the Prometheus /metrics endpoint."""

import prometheus_client
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from aegra_api.observability import metrics as metrics_module
from aegra_api.observability.metrics import registry_to_json, setup_prometheus_metrics


@pytest.fixture
def fresh_registry() -> prometheus_client.CollectorRegistry:
    """Return an isolated Prometheus registry to avoid global state leaks."""
    return prometheus_client.CollectorRegistry()


def _make_app(
    monkeypatch: pytest.MonkeyPatch,
    registry: prometheus_client.CollectorRegistry,
) -> FastAPI:
    """Create a minimal FastAPI app with Prometheus metrics enabled."""
    app = FastAPI()

    @app.get("/hello")
    def hello() -> dict[str, str]:
        return {"msg": "world"}

    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", True)
    setup_prometheus_metrics(app, registry=registry)
    return app


def test_metrics_endpoint_returns_prometheus_format(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Test that /metrics returns text in Prometheus exposition format."""
    app = _make_app(monkeypatch, fresh_registry)
    client = TestClient(app)

    # Make a request so there's something to report
    response = client.get("/hello")
    assert response.status_code == 200

    # Scrape metrics
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert "text/plain" in metrics_response.headers["content-type"]

    body = metrics_response.text
    # Should contain standard HTTP metrics from the instrumentator
    assert "http_request_duration" in body or "http_requests" in body

    explicit = client.get("/metrics", params={"format": "prometheus"})
    assert explicit.status_code == 200
    assert "text/plain" in explicit.headers["content-type"]
    assert "http_request_duration" in explicit.text or "http_requests" in explicit.text


def test_instrumented_request_on_included_router_route_does_not_500(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """A templated route added via include_router must not 500 with metrics on.

    Regression for FastAPI >=0.137: include_router puts an _IncludedRouter (no
    .path) in app.routes. The instrumentator walks app.routes and reads .path on
    a match, which raised AttributeError on every request with the old pin. Aegra
    mounts all its real routers via include_router, so this is the production path.
    """
    app = FastAPI()
    router = APIRouter()

    @router.get("/items/{item_id}")
    def get_item(item_id: str) -> dict[str, str]:
        return {"item_id": item_id}

    app.include_router(router)
    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", True)
    setup_prometheus_metrics(app, registry=fresh_registry)

    client = TestClient(app)
    response = client.get("/items/42")
    assert response.status_code == 200, f"instrumented include_router route 500'd: {response.text}"
    assert response.json() == {"item_id": "42"}


def test_reaper_counter_exposed_in_default_registry() -> None:
    """The reaper counter must live in the default registry the app exposes.

    ``setup_prometheus_metrics`` is called without a registry in production
    (main.py), so /metrics serves ``prometheus_client.REGISTRY``. All three
    outcome series are pre-created and must render even before any recovery.
    """
    body = prometheus_client.generate_latest(prometheus_client.REGISTRY).decode()

    assert "aegra_reaper_recovered_runs_total" in body
    for outcome in ("crashed_retried", "crashed_exhausted", "stuck_pending"):
        assert f'outcome="{outcome}"' in body

    reaper = registry_to_json(prometheus_client.REGISTRY)["aegra_reaper_recovered_runs_total"]
    outcomes = {sample["labels"]["outcome"] for sample in reaper["samples"] if "outcome" in sample["labels"]}
    assert outcomes == {"crashed_retried", "crashed_exhausted", "stuck_pending"}


def test_metrics_endpoint_not_exposed_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that /metrics is not available when metrics are disabled."""
    app = FastAPI()

    @app.get("/hello")
    def hello() -> dict[str, str]:
        return {"msg": "world"}

    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", False)
    setup_prometheus_metrics(app)

    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 404


def test_metrics_json_format_returns_same_http_families(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """format=json returns the registry as an object of metric families."""
    app = _make_app(monkeypatch, fresh_registry)
    client = TestClient(app)

    assert client.get("/hello").status_code == 200

    response = client.get("/metrics", params={"format": "json"})
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

    body = response.json()
    http_families = [name for name in body if name.startswith("http_")]
    assert http_families
    family = body[http_families[0]]
    assert set(family) == {"type", "help", "samples"}
    assert isinstance(family["help"], str)
    assert isinstance(family["samples"], list)
    assert family["samples"]
    sample = family["samples"][0]
    assert set(sample) >= {"name", "labels", "value"}


def test_metrics_json_format_coerces_non_finite_values(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Non-finite samples are null in the JSON body."""
    gauge = prometheus_client.Gauge(
        "aegra_test_nonfinite_http",
        "non-finite sample",
        ["kind"],
        registry=fresh_registry,
    )
    gauge.labels(kind="nan").set(float("nan"))
    gauge.labels(kind="inf").set(float("inf"))

    app = _make_app(monkeypatch, fresh_registry)
    response = TestClient(app).get("/metrics", params={"format": "json"})
    assert response.status_code == 200

    samples = {
        sample["labels"]["kind"]: sample["value"]
        for sample in response.json()["aegra_test_nonfinite_http"]["samples"]
        if sample["name"] == "aegra_test_nonfinite_http"
    }
    assert samples == {"nan": None, "inf": None}


def test_metrics_rejects_unknown_format(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """format values outside prometheus|json return 422."""
    app = _make_app(monkeypatch, fresh_registry)
    response = TestClient(app).get("/metrics", params={"format": "xml"})
    assert response.status_code == 422
