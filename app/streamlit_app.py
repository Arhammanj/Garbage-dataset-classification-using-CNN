from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.resnet_transfer import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_waste_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    classes = checkpoint["classes"]
    img_size = checkpoint["img_size"]

    model = build_model(
        num_classes=len(classes),
        freeze_backbone=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model, classes, img_size


model, classes, img_size = load_model()


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
])


def predict_image(image, top_k=3):

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)

        top_probs, top_indices = torch.topk(
            probabilities,
            k=top_k,
            dim=1
        )

    predictions = []

    for probability, index in zip(
        top_probs[0],
        top_indices[0]
    ):
        predictions.append({
            "class": classes[index.item()],
            "confidence": probability.item() * 100,
        })

    return predictions


st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="centered",
)

st.title("♻️ Waste Classification")

st.write(
    "Upload an image and the ResNet18 model will classify "
    "the type of waste."
)

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Classify Image"):

        with st.spinner("Classifying..."):

            predictions = predict_image(image)

        st.subheader("Prediction")

        top_prediction = predictions[0]

        st.success(
            f"{top_prediction['class'].upper()} "
            f"({top_prediction['confidence']:.2f}% confidence)"
        )

        st.subheader("Top 3 Predictions")

        for prediction in predictions:

            st.write(
                f"**{prediction['class']}** — "
                f"{prediction['confidence']:.2f}%"
            )