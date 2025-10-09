FROM registry.hf.space/linoyts-qwen-image-edit-2509-fast:latest

RUN pip install --no-cache-dir runpod

RUN mkdir -p /home/user/app/models

ENV HF_HOME=/home/user/app/models
ENV TRANSFORMERS_CACHE=/home/user/app/models
ENV HF_HUB_CACHE=/home/user/app/models

COPY download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py && rm /tmp/download_models.py

COPY handler.py /home/user/app/handler.py


ENV PYTHONUNBUFFERED=1
ENV ENABLE_COMPILE=false

CMD ["python", "-u", "/home/user/app/handler.py"]