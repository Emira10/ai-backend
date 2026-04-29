from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io
import torchvision.transforms as transforms

app = FastAPI()

# تحميل الموديل
model = torch.load("skin_model.pth", map_location="cpu")
model.eval()

# أسماء الكلاسات (عدليهم حسب موديلك)
classes = ["psoriasis", "eczema", "normal"]

# تجهيز الصورة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return {
        "prediction": classes[pred.item()],
        "confidence": float(confidence.item())
    }