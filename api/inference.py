import os
import hashlib
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from .settings import MODEL_URI, MODEL_CACHE_DIR, CLASS_NAMES

# Rutas de cache
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
CORE_NAME = "core_model"  # se completará con extensión detectada
WRAPPER_NAME = "multilabel_inference.keras"
CORE_PATH = None  # se setea en runtime según extensión
WRAPPER_PATH = os.path.join(MODEL_CACHE_DIR, WRAPPER_NAME)

_model = None

def _filename_from_uri(uri: str) -> str:
    # simple parse de nombre final si viene limpio; si no, usa hash
    base = os.path.basename(uri.split("?")[0])
    if base and "." in base:
        return base
    h = hashlib.sha256(uri.encode()).hexdigest()[:16]
    return f"{CORE_NAME}_{h}.bin"

def _download(uri: str, dst: str):
    print(f"⬇️ Descargando artefacto desde {uri}")
    r = requests.get(uri, stream=True, timeout=120)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print(f"✅ Descargado en {dst} ({os.path.getsize(dst)} bytes)")

def _ensure_core_available() -> str:
    global CORE_PATH
    fname = _filename_from_uri(MODEL_URI)
    CORE_PATH = os.path.join(MODEL_CACHE_DIR, fname)
    if not os.path.exists(CORE_PATH) or os.path.getsize(CORE_PATH) == 0:
        _download(MODEL_URI, CORE_PATH)
    else:
        print(f"🧩 Core encontrado en caché: {CORE_PATH}")
    return CORE_PATH

def _build_wrapper_from_core(core_model: keras.Model) -> keras.Model:
    """
    Replica tu notebook: wrapper SIN Lambda
    - Input: uint8
    - Resizing(224,224)
    - Rescaling(1/255)
    - Normalization(ImageNet)
    - core(...)
    """
    i = keras.Input(shape=(None, None, 3), dtype='uint8', name='raw_uint8_image')
    x = keras.layers.Resizing(224, 224, name='resize_224')(i)
    x = keras.layers.Rescaling(1./255, name='rescale_01')(x)
    norm = keras.layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2],
        name='imagenet_norm'
    )
    x = norm(x)
    y = core_model(x)
    return keras.Model(inputs=i, outputs=y, name='DenseNet121_Inference_NoLambda')

def _ensure_wrapper_built():
    # Si ya existe wrapper limpio, lo usamos; si no, construimos desde el core
    if os.path.exists(WRAPPER_PATH) and os.path.getsize(WRAPPER_PATH) > 0:
        print(f"🧩 Wrapper limpio en caché: {WRAPPER_PATH}")
        return

    core_path = _ensure_core_available()
    print(f"🧠 Cargando CORE (puede contener capas legacy) desde: {core_path}")

    # Intento de carga del core (permitimos safe_mode=False por compatibilidad con .h5 antiguos)
    # Nota: esto NO afecta al wrapper final (que NO usa Lambda).
    core = keras.models.load_model(core_path, compile=False, safe_mode=False)

    print("🧱 Construyendo wrapper de inferencia SIN Lambda…")
    wrapper = _build_wrapper_from_core(core)

    print(f"💾 Guardando wrapper limpio en: {WRAPPER_PATH}")
    wrapper.save(WRAPPER_PATH)

def load_or_build_model():
    """Construye (si hace falta) y carga el wrapper limpio en memoria."""
    global _model
    if _model is not None:
        return _model

    _ensure_wrapper_built()

    print("🧠 Cargando wrapper limpio a memoria…")
    _model = keras.models.load_model(WRAPPER_PATH, compile=False)
    print("✅ Modelo listo para inferencia.")
    return _model

def predict_pil_image(img: Image.Image) -> np.ndarray:
    """Recibe PIL.Image (RGB) y retorna vector de probabilidades (float[7])."""
    model = load_or_build_model()
    arr = np.expand_dims(np.array(img.convert("RGB"), dtype='uint8'), axis=0)
    preds = model.predict(arr, verbose=0)[0]
    return preds

def postprocess(preds: np.ndarray, threshold: float, top_k: int):
    probs_dict = {cls: float(preds[i]) for i, cls in enumerate(CLASS_NAMES)}
    idx = np.argsort(preds)[::-1][:top_k]
    detected = [
        {"material": CLASS_NAMES[i], "confidence": float(preds[i])}
        for i in idx if preds[i] >= threshold
    ]
    return detected, probs_dict

def map_category(label: str) -> str:
    inorganicos = {"plastic", "glass", "metal", "paper", "cardboard"}
    if label.lower() in inorganicos:
        return "Residuo Inorgánico"
    if label.lower() in {"organic"}:
        return "Residuo Orgánico"
    return "Residuo"  # fallback
