from fastapi import FastAPI, File, UploadFile
import random

app = FastAPI()

classes = ["Eczema Photos", "Lichen Planus", "gül_hast", "mantar", "sedef"]

@app.get("/")
def home():
    return {"message": "AI Backend çalışıyor"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    prediction = random.choice(classes)
    confidence = round(random.uniform(0.70, 0.95), 2)

    return {
        "prediction": prediction,
        "confidence": confidence
    }