from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_healthz_returns_ok_and_model_loaded_true():
    """
    PU-A1 – Verificación de disponibilidad del backend

    Criterios:
    - HTTP 200
    - JSON incluye status="ok"
    - JSON incluye model_loaded=True
    """
    res = client.get("/healthz")

    assert res.status_code == 200
    data = res.json()

    assert isinstance(data, dict)
    assert data.get("status") == "ok"
    assert data.get("model_loaded") is True
