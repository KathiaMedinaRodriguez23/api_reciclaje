import os
from dotenv import load_dotenv

load_dotenv()

# Dónde está el CORE (NO el wrapper roto). Ideal: *.h5 del best_phase2/final
MODEL_URI = os.getenv("MODEL_URI")  # obligatorio
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/models")
PORT = int(os.getenv("PORT", "8000"))

# Seguridad / CORS
API_KEY = os.getenv("API_KEY")  # vacío = sin auth
ALLOWED_ORIGINS = [o for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o]

# Clases y parámetros por defecto (como tu notebook)
CLASS_NAMES = ['cardboard','glass','metal','organic','paper','plastic','trash']
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "7"))
