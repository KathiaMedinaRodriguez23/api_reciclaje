from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware   # 👈 importa

from .infer import predict_image_bytes
from .classes import CLASS_NAMES

app = FastAPI(title="reciclaje-densenet121", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASS_NAMES, "num_classes": len(CLASS_NAMES)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        raise HTTPException(status_code=400, detail="Sube una imagen (jpeg/png/webp/bmp).")

    image_bytes = await file.read()
    try:
        probs = predict_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {e}")

    if probs.ndim != 1 or probs.shape[0] != len(CLASS_NAMES):
        raise HTTPException(status_code=500, detail=f"Salida inesperada del modelo: shape={probs.shape}")

    top_idx = int(probs.argmax())
    return JSONResponse({
        "class": CLASS_NAMES[top_idx],
        "score": float(probs[top_idx]),
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    })
