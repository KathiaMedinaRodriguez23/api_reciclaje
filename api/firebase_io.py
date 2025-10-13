import os
import uuid
from datetime import datetime, timezone
from typing import Tuple

import firebase_admin
from firebase_admin import credentials, storage, firestore

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_BUCKET = os.getenv("FIREBASE_BUCKET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/run/secrets/firebase_sa.js")
FIREBASE_COLLECTION = os.getenv("FIREBASE_COLLECTION", "predictions")

_app_inited = False
_db = None
_bucket = None

def init_firebase():
    global _app_inited, _db, _bucket
    if _app_inited:
        return
    if not GOOGLE_APPLICATION_CREDENTIALS or not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        raise RuntimeError("No se encontró GOOGLE_APPLICATION_CREDENTIALS o la ruta no existe.")
    if not FIREBASE_PROJECT_ID or not FIREBASE_BUCKET:
        raise RuntimeError("FIREBASE_PROJECT_ID y FIREBASE_BUCKET son obligatorios.")

    cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    firebase_admin.initialize_app(cred, {
        "projectId": FIREBASE_PROJECT_ID,
        "storageBucket": FIREBASE_BUCKET,
    })
    _db = firestore.client()
    _bucket = storage.bucket()
    _app_inited = True

def upload_image_and_get_url(content: bytes, dest_path: str) -> str:
    """
    Sube la imagen a Firebase Storage, añade un download token y retorna la URL pública (alt=media&token=...).
    dest_path: p.ej. "predictions/<uuid>.jpg"
    """
    assert _app_inited
    blob = _bucket.blob(dest_path)

    # Añade token de descarga para armar la URL pública
    token = str(uuid.uuid4())
    blob.metadata = {"firebaseStorageDownloadTokens": token}
    blob.upload_from_string(content, content_type="image/jpeg")

    # Construye URL pública estilo Firebase
    # Ojo: dest_path debe ir URL-encoded en 'o/<path>'
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
