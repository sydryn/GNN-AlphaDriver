from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.core.data_structures import VariantKey
from backend.gnn.graph_builder import FEATURE_SCHEMA
from backend.gnn.model import DriverHGT
from backend.services.alphagenome_service import MockProvider, ProviderBusy, ProviderError
from backend.services.inference import InferenceEngine, NetworkData

VARIANT = {"chr": "22", "pos": 36201698, "ref": "a", "alt": "c", "tissue": "lung", "max_points": 64}


@pytest.fixture
def client():
    with TestClient(create_app(provider=MockProvider(), engine=InferenceEngine())) as test_client:
        yield test_client


def test_health_and_cors(client):
    response = client.get("/api/v1/health")
    assert response.json() == {"status": "ok", "signal_source": "mock", "model_mode": "demo"}
    response = client.options("/api/v1/predict", headers={
        "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in client.get(
        "/api/v1/health", headers={"Origin": "https://example.com"},
    ).headers


def test_prediction_returns_provenance_score_and_track_matrices(client):
    response = client.post("/api/v1/predict", json=VARIANT)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["variant"]["chr"] == "chr22"
    assert body["variant"]["ref"] == "A"
    assert body["assembly"] == "GRCh38"
    assert body["reference_validated"] is False
    assert body["score"]["mode"] == "demo"
    assert body["score"]["calibrated"] is False
    assert 0 < body["score"]["value"] < 1
    assert len(body["attributions"]) == len(body["tracks"]) == 4
    for track in body["tracks"]:
        assert len(track["bin_edges"]) == 65
        assert track["bin_edges"][0] == body["interval"]["start"]
        assert track["bin_edges"][-1] == body["interval"]["end"]
        for ref, alt, delta in zip(track["reference"], track["alternate"], track["delta"], strict=True):
            assert [a - r for r, a in zip(ref, alt, strict=True)] == pytest.approx(delta)
    assert client.post("/api/v1/predict", json=VARIANT).json() == body


@pytest.mark.parametrize(("window", "width"), [("100kb", 131072), ("1mb", 1048576)])
def test_tracks_support_both_window_sizes(client, window, width):
    response = client.get("/api/v1/tracks", params={**VARIANT, "window": window, "pos": 1})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["interval"] == {"chromosome": "chr22", "start": 0, "end": width}
    assert "score" not in body


@pytest.mark.parametrize("changes", [
    {"pos": 0}, {"pos": True}, {"ref": "AC"}, {"alt": "A"}, {"tissue": "unknown"},
    {"window": "1kb"}, {"max_points": 4096}, {"extra": "forbidden"},
])
def test_predict_rejects_bad_requests(client, changes):
    response = client.post("/api/v1/predict", json={**VARIANT, **changes})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_variant"


def test_tracks_validate_parameters(client):
    response = client.get("/api/v1/tracks", params={**VARIANT, "chr": "99"})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_variant"


@pytest.mark.parametrize(("exception", "status"), [(ProviderError, 502), (ProviderBusy, 503)])
def test_provider_failures_are_structured_and_release_capacity(exception, status):
    class RecoveringProvider(MockProvider):
        failed = False

        def predict(self, *args):
            if not self.failed:
                self.failed = True
                raise exception("Temporary fixture failure.")
            return super().predict(*args)

    with TestClient(create_app(provider=RecoveringProvider(), engine=InferenceEngine())) as client:
        failed = client.get("/api/v1/tracks", params=VARIANT)
        assert failed.status_code == status
        assert failed.json()["code"] == ("provider_busy" if status == 503 else "provider_error")
        if status == 503:
            assert failed.headers["retry-after"] == "3"
        assert client.get("/api/v1/tracks", params=VARIANT).status_code == 200


def test_inference_admission_control():
    entered, release = Event(), Event()

    class BlockingProvider(MockProvider):
        def predict(self, *args):
            entered.set()
            assert release.wait(5)
            return super().predict(*args)

    with TestClient(create_app(provider=BlockingProvider(), engine=InferenceEngine())) as client:
        with ThreadPoolExecutor() as pool:
            request = pool.submit(client.post, "/api/v1/predict", json=VARIANT)
            try:
                assert entered.wait(5)
                assert client.get("/api/v1/health").status_code == 200
                assert client.get("/api/v1/tracks", params=VARIANT).status_code == 503
            finally:
                release.set()
            assert request.result(5).status_code == 200


def test_research_checkpoint_requires_matching_context(tmp_path):
    network = NetworkData(
        version="net-v1", genes={"g": (1.0, 0.2)}, cells={"lung": (1, 0, 0, 0)},
        expression=[("lung", "g")], variant_links={"chr22:36201698:A>C": ["g"]},
    )
    network_path, model_path = tmp_path / "network.json", tmp_path / "model.pt"
    network_path.write_text(network.model_dump_json())
    artifact = {
        "feature_schema": FEATURE_SCHEMA, "network_version": network.version,
        "signal_source": "alphagenome:0.11.0:ALL_FOLDS", "model_version": "test-v1",
        "window": "100kb", "state_dict": DriverHGT().state_dict(),
    }
    torch.save(artifact, model_path)
    engine = InferenceEngine(str(model_path), str(network_path))
    context = engine.context(VariantKey("22", 36201698, "a", "c"), "lung", "100kb")
    assert tuple(context.genes) == ("g",)
    assert context.expression == (("lung", "g"),)
    with pytest.raises(ValueError, match="indexed GRCh38"):
        create_app(provider=MockProvider(), engine=engine)

    class ReferenceCheckedFixture(MockProvider):
        name = "alphagenome"
        validates_reference = True

    with TestClient(create_app(provider=ReferenceCheckedFixture(), engine=engine)) as client:
        prediction = client.post("/api/v1/predict", json=VARIANT)
        assert prediction.status_code == 200
        assert prediction.json()["score"]["mode"] == "research"
        assert prediction.json()["reference_validated"] is True
        unsupported = client.post("/api/v1/predict", json={**VARIANT, "window": "1mb"})
        assert unsupported.status_code == 422
        assert unsupported.json()["code"] == "unsupported_context"
    torch.save({**artifact, "network_version": "different"}, model_path)
    with pytest.raises(ValueError, match="does not match"):
        InferenceEngine(str(model_path), str(network_path))
