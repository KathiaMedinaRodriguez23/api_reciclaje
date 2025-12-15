from fastapi.testclient import TestClient


def test_predictions_returns_expected_contract(monkeypatch):
    """
    PU-C1 – El endpoint /predictions devuelve el contrato esperado
    """
    import api.main as main

    # Evitar Firebase real
    monkeypatch.setattr(main, "init_firebase", lambda: None)

    # Mock de list_predictions (retorna lista vacía)
    monkeypatch.setattr(
        main,
        "list_predictions",
        lambda limit=20, start_after_iso=None: []
    )
    monkeypatch.setattr(main, "get_label_totals", lambda class_names: {
        "cardboard": 0,
        "metal": 0,
        "organic": 0,
        "paper": 2,
        "plastic": 0,
        "trash": 1,
    })

    client = TestClient(main.app)

    res = client.get("/predictions")

    assert res.status_code == 200
    data = res.json()

    assert isinstance(data, dict)
    assert "items" in data
    assert "totalsByLabel" in data
    assert "totalItemsInPage" in data
    assert "nextStartAfterIso" in data


def test_predictions_items_have_required_fields(monkeypatch):
    """
    PU-C2 – Los items del historial contienen los campos requeridos
    """
    import api.main as main

    monkeypatch.setattr(main, "init_firebase", lambda: None)

    fake_items = [
        {
            "id": "a1",
            "label": "Paper",
            "category": "Residuo Inorgánico",
            "dateIso": "2025-12-12T18:28:20Z",
            "thumbnail": "https://mock/paper.jpg",
        },
        {
            "id": "a2",
            "label": "Trash",
            "category": "Residuo",
            "dateIso": "2025-12-12T18:27:09Z",
            "thumbnail": "https://mock/trash.jpg",
        },
    ]

    monkeypatch.setattr(main, "list_predictions", lambda limit=20, start_after_iso=None: fake_items)
    monkeypatch.setattr(main, "get_label_totals", lambda class_names: {
        "cardboard": 0,
        "metal": 0,
        "organic": 0,
        "paper": 2,
        "plastic": 0,
        "trash": 1,
    })

    client = TestClient(main.app)
    res = client.get("/predictions")

    assert res.status_code == 200
    data = res.json()

    items = data.get("items")
    assert isinstance(items, list)
    assert len(items) == 2

    for it in items:
        assert "id" in it and isinstance(it["id"], str) and it["id"]
        assert "label" in it and isinstance(it["label"], str)
        assert "category" in it and isinstance(it["category"], str)
        assert "dateIso" in it and isinstance(it["dateIso"], str)
        assert "thumbnail" in it and isinstance(it["thumbnail"], str)


def test_predictions_total_items_in_page_matches_items_length(monkeypatch):
    """
    PU-C3 – totalItemsInPage coincide con la cantidad de items retornados
    """
    import api.main as main

    monkeypatch.setattr(main, "init_firebase", lambda: None)

    fake_items = [
        {
            "id": "a1",
            "label": "Paper",
            "category": "Residuo Inorgánico",
            "dateIso": "2025-12-12T18:28:20Z",
            "thumbnail": "https://mock/paper.jpg",
        },
        {
            "id": "a2",
            "label": "Trash",
            "category": "Residuo",
            "dateIso": "2025-12-12T18:27:09Z",
            "thumbnail": "https://mock/trash.jpg",
        },
        {
            "id": "a3",
            "label": "Plastic",
            "category": "Residuo Inorgánico",
            "dateIso": "2025-12-07T02:35:39Z",
            "thumbnail": "https://mock/plastic.jpg",
        },
    ]

    monkeypatch.setattr(main, "list_predictions", lambda limit=20, start_after_iso=None: fake_items)
    monkeypatch.setattr(main, "get_label_totals", lambda class_names: {
        "cardboard": 0,
        "metal": 0,
        "organic": 0,
        "paper": 2,
        "plastic": 0,
        "trash": 1,
    })

    client = TestClient(main.app)
    res = client.get("/predictions")

    assert res.status_code == 200
    data = res.json()

    items = data.get("items", [])
    assert isinstance(items, list)

    assert data.get("totalItemsInPage") == len(items)


def test_predictions_next_cursor_is_last_item_dateiso(monkeypatch):
    """
    PU-C4.1 – nextStartAfterIso coincide con el dateIso del último item
    """
    import api.main as main

    monkeypatch.setattr(main, "init_firebase", lambda: None)

    fake_items = [
        {
            "id": "a1",
            "label": "Paper",
            "category": "Residuo Inorgánico",
            "dateIso": "2025-12-12T18:28:20Z",
            "thumbnail": "https://mock/paper.jpg",
        },
        {
            "id": "a2",
            "label": "Trash",
            "category": "Residuo",
            "dateIso": "2025-12-01T19:31:20Z",  # <- último de la página
            "thumbnail": "https://mock/trash.jpg",
        },
    ]

    monkeypatch.setattr(main, "list_predictions", lambda limit=20, start_after_iso=None: fake_items)
    monkeypatch.setattr(main, "get_label_totals", lambda class_names: {
        "cardboard": 0,
        "metal": 0,
        "organic": 0,
        "paper": 2,
        "plastic": 0,
        "trash": 1,
    })

    client = TestClient(main.app)
    res = client.get("/predictions")

    assert res.status_code == 200
    data = res.json()

    assert data.get("nextStartAfterIso") == "2025-12-01T19:31:20Z"


def test_predictions_next_cursor_is_null_when_no_items(monkeypatch):
    """
    PU-C4.2 – Si no hay items, nextStartAfterIso debe ser null
    """
    import api.main as main

    monkeypatch.setattr(main, "init_firebase", lambda: None)
    monkeypatch.setattr(main, "list_predictions", lambda limit=20, start_after_iso=None: [])
    monkeypatch.setattr(main, "get_label_totals", lambda class_names: {
        "cardboard": 0,
        "metal": 0,
        "organic": 0,
        "paper": 2,
        "plastic": 0,
        "trash": 1,
    })

    client = TestClient(main.app)
    res = client.get("/predictions")

    assert res.status_code == 200
    data = res.json()

    assert data.get("items") == []
    assert data.get("nextStartAfterIso") is None


def test_predictions_totals_by_label_includes_all_classes(monkeypatch):
    """
    PU-C5 – totalsByLabel incluye todas las clases y cuenta correctamente
    """
    import api.main as main

    # Evitar Firebase real
    monkeypatch.setattr(main, "init_firebase", lambda: None)

    # Clases configuradas (ejemplo realista)
    class_names = ["cardboard", "metal", "organic", "paper", "plastic", "trash"]
    monkeypatch.setattr(main, "CLASS_NAMES", class_names, raising=False)

    # Datos simulados (solo aparecen paper y trash)
    fake_items = [
        {
            "id": "a1",
            "label": "Paper",
            "category": "Residuo Inorgánico",
            "dateIso": "2025-12-12T18:28:20Z",
            "thumbnail": "https://mock/paper.jpg",
        },
        {
            "id": "a2",
            "label": "Trash",
            "category": "Residuo",
            "dateIso": "2025-12-12T18:27:09Z",
            "thumbnail": "https://mock/trash.jpg",
        },
        {
            "id": "a3",
            "label": "Paper",
            "category": "Residuo Inorgánico",
            "dateIso": "2025-12-07T02:35:39Z",
            "thumbnail": "https://mock/paper2.jpg",
        },
    ]

    monkeypatch.setattr(main, "list_predictions", lambda limit=20, start_after_iso=None: fake_items)
    monkeypatch.setattr(main, "get_label_totals", lambda class_names: {
        "cardboard": 0,
        "metal": 0,
        "organic": 0,
        "paper": 2,
        "plastic": 0,
        "trash": 1,
    })

    client = TestClient(main.app)
    res = client.get("/predictions")

    assert res.status_code == 200
    data = res.json()

    totals = data.get("totalsByLabel")
    assert isinstance(totals, dict)

    # Todas las clases deben existir
    for cls in class_names:
        assert cls in totals

    # Conteos correctos
    assert totals["paper"] == 2
    assert totals["trash"] == 1

    # Clases sin apariciones deben ser 0
    assert totals["cardboard"] == 0
    assert totals["metal"] == 0
    assert totals["organic"] == 0
    assert totals["plastic"] == 0
