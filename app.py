# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import librosa
import matplotlib.cm as cm
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Parkinson's Disease Prediction System | Clinical XAI",
    page_icon="🧠",
    layout="wide"
)

# --- INITIALIZE HISTORY ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# =====================================================
# UI STYLING
# =====================================================
st.markdown("""
<style>
.stApp { background-color: #0B0E14; color: #E5E7EB; }
[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #374151; }
.hero-text {
    background: linear-gradient(90deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800; font-size: 3.2rem !important;
    text-align: center; margin-bottom: 5px;
}
.feature-card {
    background: #1F2937; padding: 25px; border-radius: 15px;
    border-top: 5px solid #3B82F6; transition: 0.3s;
    height: 100%;
}
h1,h2,h3,h4,p,label { color: #E5E7EB !important; }
.stButton>button {
    background: linear-gradient(45deg, #2563EB, #7C3AED);
    color: white !important; font-weight: bold; border-radius: 10px;
    border: none; height: 3.5rem; width: 100%;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODELS & UTILITIES
# =====================================================
@st.cache_resource
def load_models():
    return {
        "spiral": tf.keras.models.load_model("VGG16_Spiral_Parkinsons_Model.keras"),
        "wave": tf.keras.models.load_model("wave_model_81_25_acc.keras"),
        "voice": tf.keras.models.load_model("parkinsons_inceptionv3_spectrogram_model.keras")
    }

models = load_models()
MODEL_ACCURACY = {"spiral": 0.8667, "wave": 0.82, "voice": 0.9231}

def find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D): return layer.name
    raise ValueError("No Conv2D layer found.")

def gradcam(img_batch, model):
    layer_name = find_last_conv(model)
    grad_model = tf.keras.Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        if isinstance(preds, (list, tuple)): preds = preds[0]
        loss = preds[:, 0] if preds.shape[-1] == 1 else preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

def overlay(img, heatmap, alpha=0.4):
    heatmap_overlay = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_overlay, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

def export_record(record):
    return f"""
    MEDICAL RECORD: PARKINSON'S DISEASE PREDICTION SYSTEM
    Generated on: {record['timestamp']}
    -------------------------------------------
    PATIENT NAME: {record['name']}
    PATIENT AGE:  {record['age']}
    -------------------------------------------
    DIAGNOSTIC DATA:
    Modality:     {record['modality']}
    Result:       {record['result']}
    Confidence:   {record['confidence']}
    System Rel.:  {record['reliability']}
    -------------------------------------------
    """

# =====================================================
# AUDIO PREPROCESSING
# =====================================================
TARGET_SR, TARGET_LEN, IMG_SIZE = 8000, int(1.5 * 8000), (600, 600)

def audio_to_spectrogram(audio):
    y, sr = librosa.load(audio, sr=None)
    if sr != TARGET_SR: y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    y = librosa.util.fix_length(y, size=TARGET_LEN)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=128))
    S = 10 * np.log10(np.maximum(S, 1e-10) / np.max(S))
    S = (S - S.min()) / (S.max() - S.min())
    img = cm.magma(S)[..., :3]
    return np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(IMG_SIZE)) / 255.0

def voice_xai(img, model):
    img_batch = tf.convert_to_tensor(img[None, ...], tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_batch)
        preds = model(img_batch); loss = preds[:, 0] if preds.shape[-1] == 1 else preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, img_batch)[0]
    heatmap = tf.reduce_mean(tf.abs(grads), axis=-1).numpy()
    heatmap /= (np.percentile(heatmap, 95) + 1e-8)
    return cv2.GaussianBlur(np.clip(heatmap, 0, 1), (0, 0), sigmaX=1.2, sigmaY=1.2)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("<h1 style='color: #60A5FA;'>🧬 PD System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("📋 Patient Metadata")
    p_name = st.text_input("Patient Full Name", "Aniket Sahgal")
    p_age = st.number_input("Patient Age", 1, 120, 60)
    st.markdown("---")
    menu = st.radio("Navigation", [
        "🌟 Project Overview",
        "🌀 Spiral Diagnosis",
        "🌊 Wave Diagnosis",
        "🎤 Voice Analysis",
        "📜 History Tracker",
        "📊 Clinical Report"
    ])
    st.markdown("---")
    st.info("System v3.1 | XAI Enabled")

# =====================================================
# PAGES
# =====================================================

# --- 1. PROJECT OVERVIEW (SMALLER PICTURE) ---
if menu == "🌟 Project Overview":
    st.markdown("<h1 class='hero-text'>Parkinson's Disease Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.1rem; color:#9CA3AF; margin-bottom:20px;'>Multi-Modal Deep Learning for Early PD Biomarker Detection</p>", unsafe_allow_html=True)
    
    # Using columns to center and shrink the image
    _, img_col, _ = st.columns([1, 1.2, 1])
    with img_col:
        st.image("https://img.freepik.com/free-vector/human-brain-structure-concept_1284-18837.jpg", width=450)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='feature-card'><h4 style='color:#60A5FA;'>🔍 The Problem</h4><p style='font-size:0.9rem;'>Early-stage Parkinson's is difficult to catch. Micro-tremors in motor tasks and vocal cord jitters are early diagnostic biomarkers.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='feature-card'><h4 style='color:#A78BFA;'>⚙️ Methodology</h4><p style='font-size:0.9rem;'>Applying <b>VGG16</b> for drawings and <b>InceptionV3</b> for spectrograms to identify high-dimensional disease features.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='feature-card'><h4 style='color:#10B981;'>💡 Explainability</h4><p style='font-size:0.9rem;'><b>Grad-CAM</b> visualizations provide clinical evidence by highlighting precisely where the model detects biomarkers.</p></div>", unsafe_allow_html=True)

# --- 2. DIAGNOSTIC MODULES ---
elif menu in ["🌀 Spiral Diagnosis", "🌊 Wave Diagnosis", "🎤 Voice Analysis"]:
    config = {
        "🌀 Spiral Diagnosis": ("spiral", (128,128), ["png","jpg","jpeg"]),
        "🌊 Wave Diagnosis": ("wave", (128,128), ["png","jpg","jpeg"]),
        "🎤 Voice Analysis": ("voice", None, ["wav"])
    }
    key, size, ftypes = config[menu]
    st.title(menu)
    file = st.file_uploader(f"Upload Patient Data", type=ftypes)
    
    if file and st.button("🚀 Run Diagnosis"):
        if key == "voice":
            img = audio_to_spectrogram(file); arr = img[None, ...]
        else:
            img = Image.open(file).convert("RGB").resize(size, resample=Image.BICUBIC)
            arr = np.array(img) / 255.0; arr = arr[None, ...]

        preds = models[key].predict(arr)
        conf = float(preds[0][0]) if preds.shape[-1] == 1 else float(np.max(preds[0]))
        p_class = (1 if conf >= 0.5 else 0) if preds.shape[-1] == 1 else int(np.argmax(preds[0]))
        if preds.shape[-1] == 1 and conf < 0.5: conf = 1 - conf

        heatmap = voice_xai(img, models[key]) if key == "voice" else gradcam(arr, models[key])
        diag = "Parkinson's Detected" if p_class else "Healthy Control"
        
        # LOG HISTORY
        new_record = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": p_name, "age": p_age, "modality": menu, "result": diag, "confidence": f"{conf*100:.2f}%", "reliability": f"{MODEL_ACCURACY[key]*100:.1f}%"}
        st.session_state['history'].append(new_record)

        # UI
        st.markdown("### 🔍 XAI Visualization")
        c1, c2, c3 = st.columns(3)
        c1.image(arr[0], caption="Input Sample", use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cv2.resize(heatmap, (arr[0].shape[1], arr[0].shape[0]), interpolation=cv2.INTER_NEAREST), cmap="jet")
            ax.axis("off"); plt.tight_layout(pad=0); st.pyplot(fig, clear_figure=True)
            st.caption("AI Diagnostic Map")
        c3.image(overlay((arr[0]*255).astype(np.uint8), heatmap), caption="Scan Overlay", use_container_width=True)
        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Status", diag); r2.metric("Confidence", f"{conf*100:.2f}%"); r3.metric("Accuracy", f"{MODEL_ACCURACY[key]*100:.1f}%")
        st.download_button("📄 Download Medical Record (.txt)", export_record(new_record), f"PD_Record_{p_name}.txt")

# --- 3. HISTORY TRACKER ---
elif menu == "📜 History Tracker":
    st.title("📜 Patient Analysis History")
    if not st.session_state['history']: st.info("No records found in this session.")
    else:
        st.table(pd.DataFrame(st.session_state['history']))
        if st.button("🗑️ Clear All Records"): st.session_state['history'] = []; st.rerun()

# --- 4. CLINICAL REPORT ---
elif menu == "📊 Clinical Report":
    st.markdown("<h2 style='color: #60A5FA;'>📊 Clinical Performance & Validation</h2>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Accuracy", "87.0%", "Validated")
    m2.metric("Voice Recall", "95.0%", "Peak Sensitivity")
    m3.metric("System Split", "80/20", "Train/Test")
    st.table(pd.DataFrame({"Modality": ["Spiral Drawing", "Wave Drawing", "Voice Analysis"], "Precision": [0.88, 0.79, 0.91], "Recall": [0.85, 0.89, 0.95], "Accuracy": ["86.67%", "82.00%", "92.31%"]}).set_index("Modality"))
    st.divider()
    st.subheader("🛠 Explainability Methodology")
    st.write("Grad-CAM allows clinicians to verify that the model is looking at the correct features (like shaky line strokes or specific vocal harmonics) rather than noise.")
