FROM registry.hf.space/linoyts-qwen-image-edit-2509-fast:latest

RUN pip install --no-cache-dir runpod

ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HUB_CACHE=/app/models

RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen-Image-Edit-2509', cache_dir='/app/models'); \
    snapshot_download('lightx2v/Qwen-Image-Lightning', cache_dir='/app/models')"

COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1
ENV ENABLE_COMPILE=false

CMD ["python", "-u", "/app/handler.py"]