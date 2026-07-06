import runpod
import base64
import io
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Cargar BiRefNet una sola vez al arrancar el worker
model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
model.to("cuda").eval()
torch.set_float32_matmul_precision("high")

transformar = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def handler(job):
    entrada = job["input"]
    img_b64 = entrada.get("image")
    if not img_b64:
        return {"error": "Falta 'image' (base64) en el input"}

    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")

    tensor = transformar(img).unsqueeze(0).to("cuda")
    with torch.no_grad():
        pred = model(tensor)[-1].sigmoid().cpu()

    mascara = transforms.ToPILImage()(pred[0].squeeze()).resize(img.size)

    recorte = img.convert("RGBA")
    recorte.putalpha(mascara)

    buf = io.BytesIO()
    recorte.save(buf, format="PNG")
    return {"image": base64.b64encode(buf.getvalue()).decode()}


runpod.serverless.start({"handler": handler})
