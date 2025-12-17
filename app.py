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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="NeuroScan Pro | Explainable Parkinson’s AI",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# UI STYLING (READABLE & SAFE)
# =====================================================
st.markdown("""
<style>
.stApp { background-color: #0B0E14; color: #E5E7EB; }
[data-testid="stSidebar"] { background-color: #111827; }
h1,h2,h3,h4,p,label { color: #E5E7EB !important; }
.stButton>button {
    background: linear-gradient(45deg, #2563EB, #7C3AED);
    color: white; font-weight: bold; border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    return {
        "spiral": tf.keras.models.load_model("VGG16_Spiral_Parkinsons_Model.keras"),
        "wave": tf.keras.models.load_model("wave_model_81_25_acc.keras"),
        "voice": tf.keras.models.load_model("parkinsons_inceptionv3_spectrogram_model.keras")
    }

models = load_models()

MODEL_ACCURACY = {
    "spiral": 0.8667,
    "wave": 0.82,
    "voice": 0.9231
}

# =====================================================
# GRAD-CAM HELPERS (ROBUST)
# =====================================================
def find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

def gradcam(img_batch, model):
    layer_name = find_last_conv(model)

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if preds.shape[-1] == 1:
            loss = preds[:, 0]
        else:
            loss = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]                    # (h, w, c)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)

    # 🔥 VERY IMPORTANT: normalize softly
    heatmap /= (np.max(heatmap) + 1e-8)

    return heatmap



def overlay(img, heatmap, alpha=0.4):
    """
    img: uint8 RGB (H,W,3)
    heatmap: LOW-RES Grad-CAM (e.g. 7x7)
    """

    # 🔥 Create a SMOOTH version ONLY for overlay
    heatmap_overlay = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_CUBIC   # smooth
    )

    heatmap_overlay = np.clip(heatmap_overlay, 0, 1)
    heatmap_uint8 = np.uint8(255 * heatmap_overlay)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        img,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay





# =====================================================
# VOICE PREPROCESSING + XAI
# =====================================================
TARGET_SR = 8000
TARGET_LEN = int(1.5 * TARGET_SR)
IMG_SIZE = (600, 600)

def audio_to_spectrogram(audio):
    y, sr = librosa.load(audio, sr=None)

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # ✅ FIXED HERE
    y = librosa.util.fix_length(y, size=TARGET_LEN)

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=128))
    S = 10 * np.log10(np.maximum(S, 1e-10) / np.max(S))

    S = (S - S.min()) / (S.max() - S.min())

    img = cm.magma(S)[..., :3]
    img = Image.fromarray((img * 255).astype(np.uint8)).resize(IMG_SIZE)

    return np.array(img) / 255.0


def voice_xai(img, model):
    img_batch = tf.convert_to_tensor(img[None, ...], tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_batch)
        preds = model(img_batch)

        if preds.shape[-1] == 1:
            loss = preds[:, 0]
        else:
            loss = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, img_batch)[0]

    heatmap = tf.reduce_mean(tf.abs(grads), axis=-1).numpy()

    # 🔥 Percentile normalization
    heatmap /= (np.percentile(heatmap, 95) + 1e-8)
    heatmap = np.clip(heatmap, 0, 1)


    # 🔥 Gentle smoothing (keeps detail)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=1.2, sigmaY=1.2)

    return heatmap

def heatmap_to_rgb(heatmap):
    """
    Convert a normalized heatmap (0–1) to RGB using a colormap.
    Output is RGB uint8 → PERFECT for Streamlit.
    """
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply BLUE-dominant colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert BGR → RGB
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return heatmap_rgb





# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 🧬 NeuroScan AI")
    menu = st.radio("Select Module", [
        "🌟 Project Overview",
        "🌀 Spiral Diagnosis",
        "🌊 Wave Diagnosis",
        "🎤 Voice Analysis",
        "📊 Clinical Report"
    ])

# =====================================================
# PAGES
# =====================================================
if menu == "🌟 Project Overview":
    st.markdown("## 🧠 Early Parkinson’s Detection")
    st.write("Multi-modal deep learning with Explainable AI.")

elif menu in ["🌀 Spiral Diagnosis", "🌊 Wave Diagnosis", "🎤 Voice Analysis"]:

    config = {
        "🌀 Spiral Diagnosis": ("spiral", (128,128), ["png","jpg","jpeg"]),
        "🌊 Wave Diagnosis": ("wave", (128,128), ["png","jpg","jpeg"]),
        "🎤 Voice Analysis": ("voice", None, ["wav"])
    }

    key, size, ftypes = config[menu]
    model = models[key]

    file = st.file_uploader("Upload Sample", type=ftypes)

    if file and st.button("🚀 Run Analysis"):
        if key == "voice":
            img = audio_to_spectrogram(file)
            arr = img[None, ...]
        else:
            img = Image.open(file).convert("RGB").resize(
                size,
                resample=Image.BICUBIC
            )

            arr = np.array(img) / 255.0
            arr = arr[None, ...]

        preds = model.predict(arr)

        if preds.shape[-1] == 1:
            p_pd = float(preds[0][0])
            confidence = max(p_pd, 1-p_pd)
            pred_class = 1 if p_pd >= 0.5 else 0
        else:
            confidence = float(np.max(preds[0]))
            pred_class = int(np.argmax(preds[0]))

        if key == "voice":
            heatmap = voice_xai(img, model)
        else:
            heatmap = gradcam(arr, model)

        # ===============================
        # CREATE TWO HEATMAP VERSIONS
        # ===============================

        # 1️⃣ BLOCKY heatmap for visualization (squares visible)
        heatmap_display = cv2.resize(
            heatmap,
            (arr[0].shape[1], arr[0].shape[0]),
            interpolation=cv2.INTER_NEAREST   # 🔥 keeps square blocks
        )

        # Normalize for plotting
        heatmap_display = heatmap_display / (heatmap_display.max() + 1e-8)


        overlay_img = overlay((arr[0]*255).astype(np.uint8), heatmap)

        c1, c2, c3 = st.columns(3)
        c1.image(arr[0], caption="Input", use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))  # 🔥 smaller & stable

            ax.imshow(
                heatmap_display,
                cmap="jet",
                interpolation="nearest",  # 🔥 keeps square grid
                aspect="equal"
            )


            ax.axis("off")

            # 🔥 REMOVE all padding & margins
            plt.tight_layout(pad=0)

            st.pyplot(fig, clear_figure=True)
            st.caption("XAI Heatmap")



        c3.image(overlay_img, caption="Overlay", use_container_width=True)

        diagnosis = "Parkinson’s Detected" if pred_class else "Healthy Control"
        st.success(f"**Diagnosis:** {diagnosis}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")
        st.info(f"**Model Accuracy:** {MODEL_ACCURACY[key]*100:.2f}%")

elif menu == "📊 Clinical Report":
    st.markdown("## 📊 Model Performance")
    st.write("All models validated on held-out test sets with Explainable AI.")


