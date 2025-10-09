FROM registry.hf.space/linoyts-qwen-image-edit-2509-fast:latest

RUN pip install --no-cache-dir runpod

RUN mkdir -p /workspace/models && chmod -R 777 /workspace

ENV HF_HOME=/workspace/models
ENV TRANSFORMERS_CACHE=/workspace/models
ENV HF_HUB_CACHE=/workspace/models

RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen-Image-Edit-2509', cache_dir='/workspace/models'); \
    snapshot_download('lightx2v/Qwen-Image-Lightning', cache_dir='/workspace/models')"

COPY handler.py /workspace/handler.py

ENV PYTHONUNBUFFERED=1
ENV ENABLE_COMPILE=false

CMD ["python", "-u", "/workspace/handler.py"]
