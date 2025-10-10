import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Ruta del modelo (SavedModel) => .../Servo/1
MODEL_DIR = os.getenv(
    "MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "model", "Servo", "1")
)

# Cargar SavedModel (Keras 3 export / TF SavedModel)
# El export que nos mostraste acepta uint8 (None,None,3), hace resize+preprocess adentro.
_model = tf.saved_model.load(MODEL_DIR)
_infer = _model.signatures.get("serving_default")
if _infer is None:
    raise RuntimeError("No se encontró 'serving_default' en el SavedModel.")

def _to_uint8_rgb(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img, dtype=np.uint8)
    return np.expand_dims(arr, axis=0)  # (1,H,W,3)

def predict_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Devuelve vector de probabilidades (C,) en float32."""
    img = Image.open(io.BytesIO(image_bytes))
    x = _to_uint8_rgb(img)
    outputs = _infer(tf.constant(x))
    # Tomamos el primer tensor devuelto por la signature (nombre-agnóstico)
    first_tensor = next(iter(outputs.values()))
    probs = tf.convert_to_tensor(first_tensor).numpy().squeeze()
    return probs
