import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download

# =========================================
# Load model from Hugging Face
# =========================================
@st.cache_resource
def load_acne_model():
    model_path = hf_hub_download(
        repo_id="username/model_repo",      # TODO: replace with your repo
        filename="model.keras"
    )
    model = load_model(model_path, compile=False)
    return model

model = load_acne_model()
CLASS_NAMES = ["Mild", "Medium", "Severe"]


# =========================================
# Image Processing Functions
# =========================================
def remove_black(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > 10
    coords = np.argwhere(mask)

    if coords.size == 0:
        return img_bgr

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = img_bgr[y0:y1, x0:x1]
    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    mask_white = gray_crop <= 10
    cropped[mask_white] = [255, 255, 255]

    return cropped


def preprocess_image_safe(image_rgb, target_size=(224, 224), max_dim=1024):
    pil = Image.fromarray(image_rgb)
    w, h = pil.size

    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil = pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    img_np = np.array(pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    cropped_bgr = remove_black(img_bgr)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    preview = cv2.resize(cropped_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(cropped_rgb, target_size, interpolation=cv2.INTER_AREA).astype(np.float32)

    model_input = preprocess_input(resized)
    return model_input, preview


# =========================================
# Prediction Function
# =========================================
def predict_severity(image):
    x, preview = preprocess_image_safe(image)
    x = np.expand_dims(x, 0)

    preds = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(preds))

    label = CLASS_NAMES[top_idx]
    confidence = float(preds[top_idx] * 100)

    table = { "Class": CLASS_NAMES, 
              "Confidence (%)": [round(float(p*100), 2) for p in preds] }

    return label, confidence, preview, table


# =========================================
# Streamlit UI
# =========================================
st.title("AI Acne Severity Classifier")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original Image", width=300)

    img_np = np.array(image)

    if st.button("Predict"):
        label, conf, preview, table = predict_severity(img_np)

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: **{conf:.2f}%**")

        st.image(preview, caption="Processed Image Preview", width=300)

        st.table(table)
