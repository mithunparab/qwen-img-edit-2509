FROM registry.hf.space/linoyts-qwen-image-edit-2509-fast:latest

RUN pip install --no-cache-dir runpod

RUN mkdir -p /home/user/app/models

ENV HF_HOME=/home/user/app/models
ENV TRANSFORMERS_CACHE=/home/user/app/models
ENV HF_HUB_CACHE=/home/user/app/models

RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen-Image-Edit-2509', cache_dir='/home/user/app/models'); \
    snapshot_download('lightx2v/Qwen-Image-Lightning', cache_dir='/home/user/app/models')"

COPY handler.py /home/user/app/handler.py

ENV PYTHONUNBUFFERED=1
ENV ENABLE_COMPILE=false

CMD ["python", "-u", "/home/user/app/handler.py"]
