import importlib
from fastapi.testclient import TestClient
from io import BytesIO
from unittest.mock import MagicMock
import numpy as np


def _client_no_api_key(monkeypatch):
    """
    Crea un TestClient recargando la app con API_KEY deshabilitada.
    """
    monkeypatch.delenv("API_KEY", raising=False)

    import api.settings
    import api.main

    importlib.reload(api.settings)
    importlib.reload(api.main)

    return TestClient(api.main.app)


def _reload_main(monkeypatch, api_key=None):
    if api_key is None:
        monkeypatch.delenv("API_KEY", raising=False)
    else:
        monkeypatch.setenv("API_KEY", api_key)

    import api.settings, api.main
    import importlib
    importlib.reload(api.settings)
    importlib.reload(api.main)
    return api.main


def _client_with_api_key(monkeypatch, key="test-key"):
    """
    Crea un TestClient recargando la app con API_KEY configurada.
    """
    monkeypatch.setenv("API_KEY", key)

    import api.settings
    import api.main

    importlib.reload(api.settings)
    importlib.reload(api.main)

    return TestClient(api.main.app), key


def test_predict_rejects_non_image_file(monkeypatch):
    client = _client_no_api_key(monkeypatch)

    fake_file = BytesIO(b"esto no es una imagen")
    res = client.post(
        "/predict",
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert res.status_code == 415
    assert res.json().get("detail") == "Unsupported media type"


def test_predict_rejects_empty_file(monkeypatch):
    client = _client_no_api_key(monkeypatch)

    empty = BytesIO(b"")
    res = client.post(
        "/predict",
        files={"file": ("empty.jpg", empty, "image/jpeg")},
    )
    assert res.status_code == 400
    assert res.json().get("detail") == "Empty file"


def test_predict_rejects_invalid_image_bytes(monkeypatch):
    client = _client_no_api_key(monkeypatch)

    corrupt = BytesIO(b"not-a-real-image-content")
    res = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", corrupt, "image/jpeg")},
    )
    assert res.status_code == 400
    assert res.json().get("detail") == "Invalid image"


def test_predict_requires_api_key_when_enabled_missing(monkeypatch):
    """
    PU-B4.1 – Si API_KEY está habilitada, sin X-API-KEY debe rechazar.
    Criterios:
    - HTTP 401
    - detail: Invalid API Key
    """
    client, _key = _client_with_api_key(monkeypatch)

    # Imagen válida mínima 1x1 (PNG) para pasar validaciones previas
    # (si no pasas imagen válida, fallará antes con 400/415)
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    res = client.post(
        "/predict",
        files={"file": ("img.png", BytesIO(png_1x1), "image/png")},
    )

    assert res.status_code == 401
    assert res.json().get("detail") == "Invalid API Key"


def test_predict_requires_api_key_when_enabled_wrong(monkeypatch):
    """
    PU-B4.2 – Si API_KEY está habilitada, con X-API-KEY incorrecta debe rechazar.
    Criterios:
    - HTTP 401
    - detail: Invalid API Key
    """
    client, _key = _client_with_api_key(monkeypatch)

    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    res = client.post(
        "/predict",
        files={"file": ("img.png", BytesIO(png_1x1), "image/png")},
    )

    assert res.status_code == 401
    assert res.json().get("detail") == "Invalid API Key"


def test_predict_success_returns_label_and_probs_and_saves_doc(monkeypatch):
    """
    PU-B5 – Clasificación exitosa retorna contrato y registra la predicción

    Criterios:
    - HTTP 200
    - JSON incluye: label (str) y probs (dict)
    - label coincide con la mayor probabilidad
    - se llama save_prediction_doc con campos esperados
    """
    main = _reload_main(monkeypatch, api_key=None)

    # 1) Evitar efectos de startup (modelo/fb reales)
    monkeypatch.setattr(main, "load_or_build_model", lambda: None)
    monkeypatch.setattr(main, "init_firebase", lambda: None)

    # 2) Fijar CLASS_NAMES controladas (para que sea determinista)
    monkeypatch.setattr(
        main,
        "CLASS_NAMES",
        ["cardboard", "metal", "organic", "paper", "plastic", "trash"],
        raising=False,
    )

    # 3) Mock de inferencia: vector de probs (argmax = "plastic")
    fake_preds = np.array([0.01, 0.10, 0.05, 0.20, 0.92, 0.02], dtype=float)
    monkeypatch.setattr(main, "predict_pil_image", lambda img: fake_preds)

    # 4) Mock de utilidades externas
    monkeypatch.setattr(main, "now_iso_utc", lambda: "2025-12-14T00:00:00Z")
    monkeypatch.setattr(main, "upload_image_and_get_url", lambda content, dest: "https://mock/url.jpg")
    monkeypatch.setattr(main, "map_category", lambda label: "Residuo Inorgánico")

    save_mock = MagicMock(return_value={
        "id": "fixed-id",
        "label": "Plastic",
        "category": "Residuo Inorgánico",
        "dateIso": "2025-12-14T00:00:00Z",
        "thumbnail": "https://mock/url.jpg",
    })
    monkeypatch.setattr(main, "save_prediction_doc", save_mock)

    # 5) UUID determinista (para verificar path / doc_id)
    class _FakeUUID:
        def __str__(self):
            return "fixed-id"

    monkeypatch.setattr(main.uuid, "uuid4", lambda: _FakeUUID())

    client = TestClient(main.app)
    # PNG válido 1x1 (para pasar la validación real de PIL)
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    res = client.post(
        "/predict",
        files={"file": ("img.png", BytesIO(png_1x1), "image/png")}
    )

    assert res.status_code == 200, res.json()
    data = res.json()

    # Contrato mínimo
    assert isinstance(data.get("label"), str)
    assert isinstance(data.get("probs"), dict)

    # En tu código /predict retorna top-2
    assert len(data["probs"]) == 2

    # Label debe ser el argmax => plastic
    assert data["label"] == "plastic"

    # Verifica que se registró en Firestore (mock)
    assert save_mock.call_count == 1
    kwargs = save_mock.call_args.kwargs

    assert kwargs["doc_id"] == "fixed-id"
    assert kwargs["label"] == "Plastic"  # capitalizado en tu endpoint
    assert kwargs["category"] == "Residuo Inorgánico"
    assert kwargs["date_iso"] == "2025-12-14T00:00:00Z"
    assert kwargs["thumbnail_url"] == "https://mock/url.jpg"


def test_predict_probs_are_numeric_in_range_and_top2(monkeypatch):
    """
    PU-B6 – Validación de probabilidades (probs)
    Criterios:
    - probs es dict
    - len(probs) == 2
    - valores numéricos en [0, 1]
    - keys corresponden al top-2 real del vector
    """
    main = _reload_main(monkeypatch, api_key=None)

    # Evitar startup real
    monkeypatch.setattr(main, "load_or_build_model", lambda: None)
    monkeypatch.setattr(main, "init_firebase", lambda: None)

    # Clases controladas
    class_names = ["cardboard", "metal", "organic", "paper", "plastic", "trash"]
    monkeypatch.setattr(main, "CLASS_NAMES", class_names, raising=False)

    # Predicción controlada:
    # top1: plastic (0.91), top2: paper (0.42)
    fake_preds = np.array([0.11, 0.05, 0.02, 0.42, 0.91, 0.10], dtype=float)
    monkeypatch.setattr(main, "predict_pil_image", lambda img: fake_preds)

    # Mocks externos mínimos para que el endpoint termine
    monkeypatch.setattr(main, "now_iso_utc", lambda: "2025-12-14T00:00:00Z")
    monkeypatch.setattr(main, "upload_image_and_get_url", lambda content, dest: "https://mock/url.jpg")
    monkeypatch.setattr(main, "map_category", lambda label: "Residuo Inorgánico")
    monkeypatch.setattr(main, "save_prediction_doc", lambda **kwargs: {"ok": True})

    # UUID determinista
    class _FakeUUID:
        def __str__(self):
            return "fixed-id"
    monkeypatch.setattr(main.uuid, "uuid4", lambda: _FakeUUID())

    client = TestClient(main.app)

    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    res = client.post(
        "/predict",
        files={"file": ("img.png", BytesIO(png_1x1), "image/png")},
    )

    assert res.status_code == 200, res.json()
    data = res.json()

    probs = data.get("probs")
    assert isinstance(probs, dict)
    assert len(probs) == 2

    # Keys deben ser top-2 reales según fake_preds
    # top2 esperados: plastic y paper
    assert set(probs.keys()) == {"plastic", "paper"}

    # Valores numéricos dentro de rango
    for v in probs.values():
        assert isinstance(v, (int, float))
        assert 0.0 <= float(v) <= 1.0


def test_predict_uploads_image_with_expected_dest_path(monkeypatch):
    """
    PU-B7 – Subida de imagen con ruta esperada

    Criterios:
    - Se llama upload_image_and_get_url(content, "predictions/<uuid>.jpg")
    - <uuid> es el mismo usado como doc_id
    """
    main = _reload_main(monkeypatch, api_key=None)

    # Evitar startup real
    monkeypatch.setattr(main, "load_or_build_model", lambda: None)
    monkeypatch.setattr(main, "init_firebase", lambda: None)

    # Controlar clases y predicción
    monkeypatch.setattr(main, "CLASS_NAMES", ["cardboard", "plastic"], raising=False)
    monkeypatch.setattr(main, "predict_pil_image", lambda img: np.array([0.1, 0.9], dtype=float))

    # UUID determinista
    class _FakeUUID:
        def __str__(self):
            return "fixed-id"
    monkeypatch.setattr(main.uuid, "uuid4", lambda: _FakeUUID())

    # Mocks: upload y guardado
    upload_mock = MagicMock(return_value="https://mock/url.jpg")
    monkeypatch.setattr(main, "upload_image_and_get_url", upload_mock)

    save_mock = MagicMock(return_value={"ok": True})
    monkeypatch.setattr(main, "save_prediction_doc", save_mock)

    monkeypatch.setattr(main, "now_iso_utc", lambda: "2025-12-14T00:00:00Z")
    monkeypatch.setattr(main, "map_category", lambda label: "Residuo Inorgánico")

    client = TestClient(main.app)

    # PNG válido 1x1
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    res = client.post(
        "/predict",
        files={"file": ("img.png", BytesIO(png_1x1), "image/png")},
    )

    assert res.status_code == 200, res.json()

    # Validar llamada a upload con ruta esperada
    assert upload_mock.call_count == 1
    args, _kwargs = upload_mock.call_args
    # args = (content_bytes, dest_path)
    assert isinstance(args[0], (bytes, bytearray))
    assert args[1] == "predictions/fixed-id.jpg"

    # Validar que doc_id coincide con ese uuid (misma evidencia)
    assert save_mock.call_count == 1
    save_kwargs = save_mock.call_args.kwargs
    assert save_kwargs["doc_id"] == "fixed-id"


def test_predict_saves_required_fields(monkeypatch):
    """
    PU-B8 – Registro con campos requeridos

    Criterios:
    - save_prediction_doc se llama con:
      doc_id, label, category, date_iso, thumbnail_url
    - label se guarda capitalizada (según tu endpoint)
    """
    main = _reload_main(monkeypatch, api_key=None)

    # Evitar startup real
    monkeypatch.setattr(main, "load_or_build_model", lambda: None)
    monkeypatch.setattr(main, "init_firebase", lambda: None)

    # Control de clases y predicción (top1 = paper)
    monkeypatch.setattr(main, "CLASS_NAMES", ["paper", "trash"], raising=False)
    monkeypatch.setattr(main, "predict_pil_image", lambda img: np.array([0.8, 0.2], dtype=float))

    # UUID y fecha deterministas
    class _FakeUUID:
        def __str__(self):
            return "fixed-id"
    monkeypatch.setattr(main.uuid, "uuid4", lambda: _FakeUUID())
    monkeypatch.setattr(main, "now_iso_utc", lambda: "2025-12-14T00:00:00Z")

    # URL determinista
    monkeypatch.setattr(main, "upload_image_and_get_url", lambda content, dest: "https://mock/url.jpg")

    # Categoría esperada
    monkeypatch.setattr(main, "map_category", lambda label: "Residuo Inorgánico")

    # Mock de guardado
    save_mock = MagicMock(return_value={"ok": True})
    monkeypatch.setattr(main, "save_prediction_doc", save_mock)

    client = TestClient(main.app)

    # PNG válido 1x1
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    res = client.post(
        "/predict",
        files={"file": ("img.png", BytesIO(png_1x1), "image/png")},
        headers={"X-API-KEY": "tu_api_key_real"}
    )

    assert res.status_code == 200, res.json()

    # Verifica que save_prediction_doc fue llamado con los campos correctos
    assert save_mock.call_count == 1
    kwargs = save_mock.call_args.kwargs

    assert kwargs["doc_id"] == "fixed-id"
    assert kwargs["label"] == "Paper"  # tu endpoint hace top_label.capitalize()
    assert kwargs["category"] == "Residuo Inorgánico"
    assert kwargs["date_iso"] == "2025-12-14T00:00:00Z"
    assert kwargs["thumbnail_url"] == "https://mock/url.jpg"
