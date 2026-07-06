FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN pip install --no-cache-dir runpod transformers timm einops kornia pillow

# Descargar el modelo BiRefNet dentro de la imagen (evita esperas al arrancar)
RUN python -c "from transformers import AutoModelForImageSegmentation as M; M.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)"

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
