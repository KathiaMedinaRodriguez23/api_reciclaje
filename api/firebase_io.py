import os
import uuid
from datetime import datetime, timezone
from typing import Optional, List

import google.cloud.firestore as gcf            # <— cliente google-cloud-firestore
import firebase_admin
from firebase_admin import credentials, storage, firestore as admin_fs  # <— admin firestore con alias

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_BUCKET = os.getenv("FIREBASE_BUCKET")
# FIX: extensión .json
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/run/secrets/firebase_sa.json")
FIREBASE_COLLECTION = os.getenv("FIREBASE_COLLECTION", "predictions")

_app_inited = False
_db: gcf.Client | None = None
_bucket = None

def init_firebase():
    global _app_inited, _db, _bucket
    if _app_inited:
        return
    if not GOOGLE_APPLICATION_CREDENTIALS or not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        raise RuntimeError(f"No se encontró GOOGLE_APPLICATION_CREDENTIALS o la ruta no existe: {GOOGLE_APPLICATION_CREDENTIALS}")
    if not FIREBASE_PROJECT_ID or not FIREBASE_BUCKET:
        raise RuntimeError("FIREBASE_PROJECT_ID y FIREBASE_BUCKET son obligatorios.")

    cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    firebase_admin.initialize_app(cred, {
        "projectId": FIREBASE_PROJECT_ID,
        "storageBucket": FIREBASE_BUCKET,
    })
    _db = admin_fs.client()     # client de admin
    _bucket = storage.bucket()
    _app_inited = True

def upload_image_and_get_url(content: bytes, dest_path: str) -> str:
    """
    Sube la imagen a Firebase Storage, añade un download token y retorna la URL pública (alt=media&token=...).
    dest_path: p.ej. "predictions/<uuid>.jpg"
    """
    assert _app_inited
    blob = _bucket.blob(dest_path)

    token = str(uuid.uuid4())
    blob.metadata = {"firebaseStorageDownloadTokens": token}
    blob.upload_from_string(content, content_type="image/jpeg")

    from urllib.parse import quote
    encoded = quote(dest_path, safe="")
    return f"https://firebasestorage.googleapis.com/v0/b/{FIREBASE_BUCKET}/o/{encoded}?alt=media&token={token}"

def save_prediction_doc(doc_id: str, label: str, category: str, date_iso: str, thumbnail_url: str):
    """
    Crea/actualiza un doc en Firestore con el shape pedido.
    """
    assert _app_inited
    doc = {
        "id": doc_id,
        "label": label,
        "category": category,
        "dateIso": date_iso,
        "thumbnail": thumbnail_url,
    }
    _db.collection(FIREBASE_COLLECTION).document(doc_id).set(doc)
    return doc

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def list_predictions(limit: int = 20, start_after_iso: Optional[str] = None) -> list[dict]:
    """
    Lista docs ordenados por 'dateIso' DESC.
    - start_after_iso: pasar el último dateIso de la página anterior (string ISO8601)
    """
    assert _app_inited, "init_firebase() no fue llamado"
    q = _db.collection(FIREBASE_COLLECTION).order_by("dateIso", direction=admin_fs.Query.DESCENDING)
    if start_after_iso:
        q = q.start_after(start_after_iso)
    q = q.limit(max(1, min(100, int(limit))))

    results: list[dict] = []
    for snap in q.stream():
        data = snap.to_dict() or {}
        results.append({
            "id": data.get("id") or snap.id,
            "label": data.get("label", ""),
            "category": data.get("category", ""),
            "dateIso": data.get("dateIso", ""),
            "thumbnail": data.get("thumbnail", ""),
        })
    return results
