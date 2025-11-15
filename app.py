# app.py
import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download

# ==============================
# 📌 Utility: Load model (cached)
# ==============================
@st.cache_resource(show_spinner=True)
def load_acne_model():
    repo_id = "ahsanatiq98/AI-Acne-Grading"  
    filename = "model.keras"                 
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = load_model(model_path, compile=False)
    return model

model = load_acne_model()

CLASS_NAMES = ["Mild", "Medium", "Severe"]

# ==============================
# 📌 Image – Processing Functions
# ==============================
import cv2  # Keep after checking for system libs

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

def preprocess_image_safe(image_rgb, target_size=(224,224), max_dim=1024):
    pil = Image.fromarray(image_rgb)
    w, h = pil.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil = pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    img_np = np.array(pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cropped_bgr = remove_black(img_bgr)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    preview = Image.fromarray(cv2.resize(cropped_rgb, (256,256), interpolation=cv2.INTER_AREA))
    resized_for_model = cv2.resize(cropped_rgb, target_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    model_input = preprocess_input(resized_for_model)
    return model_input, preview

# ==============================
# 📌 Prediction Function
# ==============================
def predict_severity_safe(image):
    if image is None:
        return "No image provided", None, []
    x, preview_pil = preprocess_image_safe(image)
    x = np.expand_dims(x, 0)
    preds = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    label = CLASS_NAMES[top_idx]
    confidence = float(preds[top_idx] * 100.0)
    prob_table = [[CLASS_NAMES[i], round(float(preds[i]*100.0), 2)] for i in range(len(CLASS_NAMES))]
    result_md = f"**Severity:** {label}  \n**Confidence:** {confidence:.2f}%"
    return result_md, preview_pil, prob_table

# ==============================
# 📌 Streamlit UI
# ==============================
st.title("AI Acne Severity Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    result_md, preview_img, prob_table = predict_severity_safe(image)
    st.markdown(result_md)
    if preview_img:
        st.image(preview_img, caption="Processed Preview", use_column_width=True)
    st.table(prob_table)
else:
    st.info("Please upload an image to proceed.")

# You could optionally add streamlit button, but file uploader is enough
