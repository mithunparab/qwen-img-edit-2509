FROM registry.hf.space/linoyts-qwen-image-edit-2509-fast:latest

RUN pip install --no-cache-dir runpod

RUN mkdir -p /home/user/app/models

ENV HF_HOME=/home/user/app/models
ENV TRANSFORMERS_CACHE=/home/user/app/models
ENV HF_HUB_CACHE=/home/user/app/models

RUN python3 -c "from huggingface_hub import snapshot_download; \
    print('Checking if base model exists...'); \
    try: \
    snapshot_download('Qwen/Qwen-Image-Edit-2509', cache_dir='/home/user/app/models', local_files_only=True); \
    print('Base model found in image!'); \
    except: \
    print('Downloading base model...'); \
    snapshot_download('Qwen/Qwen-Image-Edit-2509', cache_dir='/home/user/app/models'); \
    print('Downloading LoRA...'); \
    snapshot_download('lightx2v/Qwen-Image-Lightning', \
    cache_dir='/home/user/app/models', \
    allow_patterns=['Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors', '*.json', 'README.md'])"

COPY handler.py /home/user/app/handler.py

ENV PYTHONUNBUFFERED=1
ENV ENABLE_COMPILE=false

CMD ["python", "-u", "/home/user/app/handler.py"]