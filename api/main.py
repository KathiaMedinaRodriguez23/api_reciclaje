import io, uuid
from .firebase_io import init_firebase, upload_image_and_get_url, save_prediction_doc, now_iso_utc
from .inference import map_category

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .settings import (
    PORT, API_KEY, ALLOWED_ORIGINS,
    DEFAULT_THRESHOLD, DEFAULT_TOP_K, CLASS_NAMES
)
from .inference import (
    load_or_build_model, predict_pil_image, postprocess
)

app = FastAPI(title="Recycling Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def require_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.on_event("startup")
def _startup():
    # Construye/carga el modelo
    load_or_build_model()
    # Inicializa Firebase (Storage + Firestore)
    init_firebase()


@app.get("/healthz")
def healthz():
    # Si load_or_build_model falla, FastAPI ya no arranca; acá devolvemos ok
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", dependencies=[Depends(require_api_key)] if API_KEY else [])
async def predict(
    file: UploadFile = File(...),
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 1) Inferencia
    preds = predict_pil_image(img)
    probs_dict = {cls: float(preds[i]) for i, cls in enumerate(CLASS_NAMES)}
    top_label = max(probs_dict, key=probs_dict.get)

    # 2) Subir imagen al Storage
    uid = str(uuid.uuid4())
    dest_path = f"predictions/{uid}.jpg"
    thumb_url = upload_image_and_get_url(content, dest_path)

    # 3) Registrar en Firestore
    date_iso = now_iso_utc()
    category = map_category(top_label)
    doc = save_prediction_doc(
        doc_id=uid,
        label=top_label.capitalize(),     # "Plastico"
        category=category,                # "Residuo Inorgánico"
        date_iso=date_iso,                # "2025-12-23T14:05:00Z"
        thumbnail_url=thumb_url
    )

    # 4) Respuesta (como pediste)
    return {
        "label": doc["label"],
        "probs": probs_dict
    }


