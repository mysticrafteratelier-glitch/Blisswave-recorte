FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN pip install --no-cache-dir runpod transformers timm einops kornia pillow

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
