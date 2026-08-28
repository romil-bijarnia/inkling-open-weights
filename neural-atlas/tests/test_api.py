from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from safetensors.numpy import save_file

import neural_atlas.app as app_module
from neural_atlas.model_runtime import ModelRuntime


def test_checkpoint_api_without_downloading_a_model(tmp_path, monkeypatch):
    matrix = np.array([[1.0, -2.0, 3.5], [4.25, 5.0, -6.5]], dtype=np.float32)
    save_file({"probe.weight": matrix}, tmp_path / "model.safetensors")
    monkeypatch.setattr(
        app_module,
        "runtime",
        ModelRuntime(model_id=str(tmp_path), device="cpu", dtype="float32"),
    )

    with TestClient(app_module.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["exactWeightAtlas"] is True

        indexed = client.post("/api/checkpoint/index")
        assert indexed.status_code == 200
        assert indexed.json()["tensorCount"] == 1

        tensors = client.get("/api/tensors", params={"q": "probe"})
        assert tensors.status_code == 200
        assert tensors.json()["items"][0]["name"] == "probe.weight"

        tile = client.get(
            "/api/tile",
            params={
                "name": "probe.weight",
                "rowStart": 0,
                "rowCount": 2,
                "columnStart": 0,
                "columnCount": 3,
                "targetRows": 2,
                "targetColumns": 3,
            },
        )
        assert tile.status_code == 200
        assert tile.json()["sourceValues"] == 6
        assert tile.json()["cells"][1][2]["mean"] == -6.5

        scalar = client.get(
            "/api/scalar",
            params={"name": "probe.weight", "matrixRow": 0, "matrixColumn": 1},
        )
        assert scalar.status_code == 200
        payload = scalar.json()
        assert payload["indices"] == [0, 1]
        assert payload["value"] == -2.0
        assert payload["rawHex"] == matrix[0, 1].tobytes().hex()
        assert payload["exact"] is True
