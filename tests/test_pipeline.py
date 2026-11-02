import io
import subprocess
import sys

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from sonarseg.api import app
from sonarseg.model import UNet
from sonarseg.pipeline import gate

PARAMS = {"min_test_miou": 0.6, "min_margin_over_baseline": 0.1}


def test_gate_passes_good_model():
    gate({"unet_miou": 0.8, "baseline_miou": 0.4}, PARAMS)


@pytest.mark.parametrize("unet,base", [(0.5, 0.2), (0.7, 0.65)])
def test_gate_blocks_weak_model(unet, base):
    with pytest.raises(SystemExit):
        gate({"unet_miou": unet, "baseline_miou": base}, PARAMS)


def test_model_output_shape():
    assert UNet(8)(torch.zeros(2, 1, 64, 64)).shape == (2, 6, 64, 64)


@pytest.fixture
def client(tmp_path, monkeypatch):
    weights = tmp_path / "m.pt"
    torch.save({"state": UNet(8).state_dict(), "base": 8, "version": "test"}, weights)
    monkeypatch.setenv("MODEL_PATH", str(weights))
    with TestClient(app) as test_client:
        yield test_client


def test_simulate_endpoint(client):
    body = client.get("/v1/simulate?seed=3").json()
    assert 0 <= body["miou"] <= 1
    assert sum(body["coverage_percent"].values()) == pytest.approx(100, abs=0.1)


def test_rejects_bad_parameters(client):
    assert client.get("/v1/simulate?speckle_looks=0").status_code == 422


def test_upload_warns_about_simulated_training(client):
    buffer = io.BytesIO()
    Image.new("L", (200, 90), 80).save(buffer, "PNG")
    response = client.post("/v1/segment", files={"file": ("a.png", buffer.getvalue())})
    assert response.status_code == 200 and "simulated" in response.json()["warning"]


def test_upload_rejects_non_image(client):
    response = client.post("/v1/segment", files={"file": ("a.png", b"nope")})
    assert response.status_code == 415


def test_api_does_not_import_training_tools():
    """mlflow is a dev dependency, so importing it would crash the production image."""
    code = "import sys, sonarseg.api; sys.exit('mlflow' in sys.modules or 'yaml' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": "src"}).returncode == 0
