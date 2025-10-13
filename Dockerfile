FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY api/ /app/api/
RUN pip install --no-cache-dir -r /app/api/requirements.txt

EXPOSE 8000
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
