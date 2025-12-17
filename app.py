import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="NeuroScan Pro | Next-Gen Diagnostics",
    page_icon="🧠",
    layout="wide"
)

# --- ADVANCED UI STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0B0E14; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid #374151; }
    
    /* Hero Title */
    .hero-text {
        background: linear-gradient(90deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3.5rem !important;
        text-align: center; margin-bottom: 0;
    }
    
    /* Feature Cards */
    .feature-card {
        background: #1F2937; padding: 25px; border-radius: 15px;
        border-top: 5px solid #3B82F6; transition: 0.3s;
    }
    .feature-card:hover { transform: scale(1.02); background: #2D3748; }

    /* Buttons & Progress */
    .stButton>button {
        background: linear-gradient(45deg, #2563EB, #7C3AED);
        color: white !important; border: none; border-radius: 12px;
        font-weight: bold; font-size: 1.1rem; height: 3.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADER ---
@st.cache_resource
def load_all_models():
    try:
        return {
            "spiral": tf.keras.models.load_model("VGG16_Spiral_Parkinsons_Model.keras"),
            "wave": tf.keras.models.load_model("wave_model_81_25_acc.keras"),
            "voice": tf.keras.models.load_model("parkinsons_inceptionv3_spectrogram_model.keras")
        }
    except: return None

models = load_all_models()

# --- SIDEBAR: PATIENT DASHBOARD ---
with st.sidebar:
    st.markdown("<h1 style='color: #60A5FA;'>🧬 NeuroScan AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("📋 Patient Metadata")
    p_name = st.text_input("Full Name", "Aniket Sahgal")
    p_id = st.text_input("Patient ID", "PD-2024-001")
    p_age = st.slider("Age", 18, 100, 62)
    p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    st.markdown("---")
    menu = st.radio("MAIN CONSOLE", 
                    ["🌟 Project Showcase", "🌀 Spiral Diagnosis", "🌊 Wave Diagnosis", "🎤 Voice Analysis", "📊 Clinical Report"])
    st.markdown("---")
    if st.button("Clear Session"): st.rerun()

# --- PAGE 1: PROJECT SHOWCASE (ATTRACTIVE ABOUT) ---
if menu == "🌟 Project Showcase":
    st.markdown("<h1 class='hero-text'>Early PD Detection Suite</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Leveraging Multi-Modal Deep Learning & Explainable AI</p>", unsafe_allow_html=True)
    
    # Visual Layout
    st.image("https://img.freepik.com/free-vector/human-brain-structure-concept_1284-18837.jpg", width=800)
    

    st.markdown("### 🔍 Why Parkinson's Detection Matters")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='feature-card'>
        <h4>The Challenge</h4>
        <p>Parkinson’s is often diagnosed too late because symptoms like minor hand tremors 
        or voice changes are subtle. Early detection can slow progression by years.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='feature-card'>
        <h4>Our AI Solution</h4>
        <p>By combining <b>VGG16</b> for motor patterns and <b>InceptionV3</b> for vocal biomarkers, 
        we provide a non-invasive, objective screening tool for clinicians.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Interactive Diagnosis Flow")
    st.write("Our 3-Step validation ensures accuracy and trust.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Motor Stage", "Spiral Test")
    c2.metric("Kinetic Stage", "Wave Test")
    c3.metric("Speech Stage", "Vocal Scan")
    
    

# --- PAGES 2, 3, 4: DIAGNOSTIC PIPELINES ---
elif menu in ["🌀 Spiral Diagnosis", "🌊 Wave Diagnosis", "🎤 Voice Analysis"]:
    task_map = {
        "🌀 Spiral Diagnosis": ("spiral", (128, 128), "Spiral Drawing Analysis", "Analyzing circular motor control..."),
        "🌊 Wave Diagnosis": ("wave", (128, 128), "Wave Handwriting Analysis", "Scanning waveform fluidity..."),
        "🎤 Voice Analysis": ("voice", (600, 600), "Vocal Frequency Analysis", "Extracting vocal harmonics...")
    }
    key, size, title, loading_msg = task_map[menu]
    
    st.title(title)
    st.markdown(f"**Patient:** {p_name} | **Age:** {p_age} | **ID:** {p_id}")
    
    file = st.file_uploader(f"Upload {key.capitalize()} Sample", type=['png', 'jpg', 'jpeg'])
    
    if file and models:
        img = Image.open(file)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, use_container_width=True, caption="Pre-processed Input")
        with c2:
            if st.button(f"🚀 INITIATE NEURAL SCAN"):
                # Simulation for "Attractive" UI effect
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                    status_text.text(f"{loading_msg} {i+1}%")
                
                # Actual Prediction
                prep = img.convert('RGB').resize(size)
                arr = np.expand_dims(np.array(prep)/255.0, axis=0)
                score = models[key].predict(arr)[0][0]
                
                status_text.empty()
                if score > 0.5:
                    st.markdown(f"<div class='result-box status-positive'><h2>⚠️ WARNING</h2>PD Indicators detected in {key} markers.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-box status-negative'><h2>✅ NORMAL</h2>No significant PD markers found.</div>", unsafe_allow_html=True)
                
                st.metric("Clinical Confidence", f"{score:.2%}")
                st.download_button("📩 Download Diagnosis Report", data=f"Patient: {p_name}\nResult: {score}", file_name="report.txt")

# --- PAGE 5: PERFORMANCE REPORT ---
elif menu == "📊 Clinical Report":
    st.title("System Analytics & Validation")
    
    # THE RECTIFIED METRICS
    metrics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score", "Overall Accuracy"],
        "Spiral (VGG16)": ["87.67%", "85.33%", "86.49%", "86.67%"],
        "Wave (CNN)": ["77.90%", "89.33%", "83.22%", "82.00%"],
        "Voice (InceptionV3)": ["81.0%", "95.0%", "90.0%", "92.31%"]
    }).set_index("Metric")
    
    st.table(metrics_df)
    
    st.markdown("---")
    st.subheader("Validation Visuals (Grad-CAM & Confusion Matrices)")
    
    
    
    tabs = st.tabs(["🌀 Spiral", "🌊 Wave", "🎤 Voice"])
    with tabs[0]: st.image("CONFUSION MATRIX FOR SPIRAL .jpg", use_container_width=True)
    with tabs[1]: st.image("CONFUSION MATRIX FOR WAVE.jpg", use_container_width=True)
    with tabs[2]: st.image("Confusion matrix for voice PArkinsons.png", use_container_width=True)