FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_DEACTIVATE_ASYNC_LOAD=1

# CPU-версия torch (легковесная), затем остальные зависимости
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && (pip install --no-cache-dir lightfm || true)

COPY . .

EXPOSE 8000

# при первом запуске скачает датасеты и обучит модели (~15-25 мин), затем поднимет сервер
CMD ["bash", "-c", "if [ ! -f data/processed/embeddings.npy ]; then python pipeline.py; fi; python -m app.server --host 0.0.0.0 --port 8000"]
