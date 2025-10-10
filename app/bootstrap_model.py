import os
import sys
import shutil
import zipfile
from pathlib import Path

def find_servo_one(root: Path) -> Path | None:
    # Busca una carpeta que termine exactamente en Servo/1
    for p in root.rglob("*"):
        if p.is_dir() and p.name == "1" and p.parent.name == "Servo":
            return p
    return None

def main():
    file_id = os.getenv("GDRIVE_FILE_ID", "").strip()
    target_dir = Path(os.getenv("MODEL_BASE_DIR", str(Path(__file__).parent / "model")))
    target_dir.mkdir(parents=True, exist_ok=True)

    if not file_id:
        print("GDRIVE_FILE_ID no está definido. Abortando.", file=sys.stderr)
        sys.exit(1)

    zip_path = target_dir / "model.zip"
    # Descarga con gdown (maneja confirm token >100MB)
    try:
        import gdown  # instalado en requirements
    except Exception as e:
        print(f"gdown no disponible: {e}", file=sys.stderr)
        sys.exit(1)

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Descargando modelo desde: {url}")
    gdown.download(url, str(zip_path), quiet=False)

    # Limpia carpeta target antes de extraer (excepto el zip)
    for item in target_dir.iterdir():
        if item.name != "model.zip":
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    # Extrae
    print(f"Extrayendo {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)

    # Detecta Servo/1
    servo_one = find_servo_one(target_dir)
    if not servo_one:
        print("No se encontró carpeta 'Servo/1' tras extraer el ZIP. "
              "Asegúrate de que el ZIP contenga el SavedModel exportado en '.../Servo/1'.",
              file=sys.stderr)
        sys.exit(1)

    # Valor final de MODEL_DIR = <...>/model/Servo/1
    print(f"OK: encontrado SavedModel en: {servo_one}")
    # No movemos nada: infer.py por defecto usa .../model/Servo/1
    # Si el 'Servo/1' quedó en subcarpetas, lo dejamos donde está; la ruta por defecto apunta a app/model/Servo/1,
    # así que creamos ese path si no coincide y hacemos un symlink o copia.

    desired = target_dir / "Servo" / "1"
    if servo_one != desired:
        desired.parent.mkdir(parents=True, exist_ok=True)
        if desired.exists():
            if desired.is_symlink() or desired.is_file():
                desired.unlink()
            else:
                shutil.rmtree(desired)
        try:
            # Intentar symlink (si el FS lo permite)
            os.symlink(servo_one, desired, target_is_directory=True)
            print(f"Creado symlink: {desired} -> {servo_one}")
        except Exception:
            # Fallback: copiar (más lento, pero seguro)
            shutil.copytree(servo_one, desired)
            print(f"Copiado modelo a: {desired}")

    print("Bootstrap del modelo completado.")

if __name__ == "__main__":
    main()
