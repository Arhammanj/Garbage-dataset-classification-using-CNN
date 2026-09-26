from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18
from torchvision import transforms


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_waste_classifier.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="centered"
)


# ============================================================
# BUILD RESNET18 MODEL
# ============================================================

def build_model(
    num_classes: int,
    freeze_backbone: bool = True
):

    model = resnet18(
        weights=None
    )

    if freeze_backbone:

        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    classes = checkpoint["classes"]

    img_size = checkpoint["img_size"]

    model = build_model(
        num_classes=len(classes),
        freeze_backbone=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    return model, classes, img_size


model, classes, img_size = load_model()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (img_size, img_size)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(
    image,
    top_k=3
):

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

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

            "class": classes[
                index.item()
            ],

            "confidence": (
                probability.item() * 100
            )
        })

    return predictions


# ============================================================
# TITLE
# ============================================================

st.title(
    "♻️ Waste Classification"
)

st.write(
    "Upload an image and our ResNet18 "
    "deep learning model will classify "
    "the type of waste."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "Model Information"
):

    st.write(
        "**Model:** ResNet18"
    )

    st.write(
        "**Dataset Classes:** 10"
    )

    st.write(
        "**Test Accuracy:** 84.94%"
    )

    st.write(
        "**Device:**",
        DEVICE
    )


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(

    "Upload a waste image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# IMAGE DISPLAY
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # ========================================================
    # CLASSIFY BUTTON
    # ========================================================

    if st.button(
        "🔍 Classify Image",
        type="primary"
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            predictions = predict_image(
                image,
                top_k=3
            )


        # ====================================================
        # MAIN PREDICTION
        # ====================================================

        top_prediction = predictions[0]

        st.subheader(
            "Prediction"
        )

        st.success(

            f"{top_prediction['class'].upper()} "
            f"— "
            f"{top_prediction['confidence']:.2f}% "
            f"confidence"

        )


        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        st.subheader(
            "Top 3 Predictions"
        )

        for prediction in predictions:

            class_name = (
                prediction["class"]
            )

            confidence = (
                prediction["confidence"]
            )

            st.write(

                f"**{class_name.capitalize()}** "
                f"— "
                f"{confidence:.2f}%"

            )

            st.progress(
                min(
                    int(confidence),
                    100
                )
            )


else:

    st.info(
        "👆 Upload an image above "
        "to begin classification."
    )