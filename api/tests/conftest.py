# tests/conftest.py
import importlib
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def fresh_client(monkeypatch):
    """
    Devuelve un TestClient con una instancia fresca de la app,
    evitando contaminación por reloads previos.
    """
    # Asegura que API_KEY esté deshabilitada para estos tests
    monkeypatch.delenv("API_KEY", raising=False)

    import api.settings
    import api.main

    importlib.reload(api.settings)
    importlib.reload(api.main)

    return TestClient(api.main.app)
