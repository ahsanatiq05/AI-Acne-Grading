# app.py
import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download
import cv2

# =========================
# GUI
# =========================
st.markdown("""
<style>

html, body {
    background-color: #0d1117 !important;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.app-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

.app-subtitle {
    text-align: center;
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 25px;
}

.upload-card {
    border-radius: 16px;
    padding: 25px;
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.1);
}

.result-box {
    padding: 20px;
    border-radius: 16px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #4ade80, #22d3ee);
}

.info-card {
    border-radius: 12px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==================================
# 📌 Load Model (Streamlit Cached)
# ==================================
@st.cache_resource(show_spinner=True)
def load_acne_model():
    repo_id = "ahsanatiq98/AI-Acne-Grading" 
    filename = "model.keras"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_model(model_path, compile=False)

model = load_acne_model()
CLASS_NAMES = ["Mild", "Medium", "Severe"]


# ====================================
# 📌 Preprocessing & Cropping
# ====================================
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
    cropped[gray_crop <= 10] = [255, 255, 255]
    return cropped


def preprocess_image_safe(image_rgb, target_size=(224,224)):
    img_np = np.array(image_rgb)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cropped_bgr = remove_black(img_bgr)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    preview = Image.fromarray(cv2.resize(cropped_rgb, (256,256)))
    resized = cv2.resize(cropped_rgb, target_size).astype(np.float32)
    model_input = preprocess_input(resized)
    return model_input, preview


# ====================================
# 📌 Prediction
# ====================================
def predict_severity_safe(image):
    x, preview_pil = preprocess_image_safe(image)
    x = np.expand_dims(x, 0)
    preds = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    conf = float(preds[idx] * 100)

    table = {CLASS_NAMES[i]: float(preds[i]*100) for i in range(len(CLASS_NAMES))}
    return label, conf, preview_pil, table


# =========================
# 🌟 App Title
# =========================
st.markdown("<h1 class='app-title'>AI Acne Severity Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Deep Learning powered medical image grading (EfficientNet-B0)</p>", unsafe_allow_html=True)

tabs = st.tabs(["🔍 Prediction", "ℹ️ How It Works", "🤖 Model Info"])

# =========================
# 🔍 Prediction Tab UI
# =========================
with tabs[0]:
    st.markdown("<div class='upload-card'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            label, confidence, preview, probs = predict_severity_safe(image)

        st.markdown("<br><div class='result-box'>", unsafe_allow_html=True)

        st.markdown(f"### 🧾 Prediction: **{label}**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        # Confidence bar chart
        st.markdown("#### 🔎 Confidence Breakdown")
        st.progress(int(confidence))

        st.bar_chart(probs)

        st.markdown("#### Processed Preview")
        st.image(preview, width=250)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Please upload an image to start.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# ℹ️ How It Works Tab
# =========================
with tabs[1]:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("""
### 🔬 Pipeline Overview  
Your image goes through:

1. **Preprocessing**
   - Removes black borders  
   - Crops to region of interest  
   - Cleans artifacts  
2. **Resizing & Normalization**
3. **EfficientNet-B0 Forward Pass**
4. **Softmax Prediction**
5. **Confidence Visualization**

This ensures stable and accurate severity grading even on external images.
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# 🤖 Model Info Tab
# =========================
with tabs[2]:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("""
### ⚙️ Model Architecture  
- EfficientNet-B0 Backbone  
- Custom Dense Head  
- 30,000-image augmented dataset  
- 94.01% validation accuracy  

### 🗂️ Classes  
- **Mild**  
- **Medium**  
- **Severe**

### 📥 Model  
Downloaded directly from Hugging Face Hub.
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.markdown("""
<br><br>
<p style='text-align:center; color: #6b7280'>
Made with ❤️ using Streamlit · EfficientNet · Hugging Face
</p>
""", unsafe_allow_html=True)
