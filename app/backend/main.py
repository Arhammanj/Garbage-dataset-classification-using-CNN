from pathlib import Path
import io

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from src.models.resnet_transfer import build_model


# -------------------------
# Configuration
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_waste_classifier.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------
# Load trained model
# -------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

classes = checkpoint["classes"]
img_size = checkpoint["img_size"]

model = build_model(
    num_classes=len(classes),
    freeze_backbone=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# -------------------------
# Preprocessing
# -------------------------

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
])


# -------------------------
# FastAPI
# -------------------------

app = FastAPI(
    title="Waste Classification API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Waste Classification API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "ResNet18",
        "device": str(DEVICE),
        "classes": classes,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    # Convert to PIL image
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Preprocess
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    # Inference
    with torch.no_grad():

        logits = model(image_tensor)

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        top_probs, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1,
        )

    # Build top-3 response
    predictions = []

    for probability, index in zip(
        top_probs[0],
        top_indices[0],
    ):
        predictions.append({
            "class": classes[index.item()],
            "confidence": round(
                probability.item() * 100,
                2,
            ),
        })

    return {
        "filename": file.filename,
        "predictions": predictions,
    }