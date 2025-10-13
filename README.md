# Recycling Inference API

### 1️⃣ Variables de entorno
Copia `.env.example` a `.env` y edita `MODEL_URI`.

### 2️⃣ Ejecutar local
```bash
uvicorn api.main:app --reload
````

## Ejecuta el proyecto en local

### Windows

Para ejecutar el proyecto en Windows, utiliza el siguiente comando en PowerShell:

```powershell
docker compose up --build -d ; if($?) {docker system prune -a -f --volumes}
```

### Linux

Para ejecutar el proyecto en Linux, utiliza el siguiente comando en la terminal:

```bash
docker compose up --build -d && docker system prune -a -f --volumes
```