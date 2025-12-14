import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import plotly.graph_objects as go

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Eye Disease Detection AI",
    page_icon="👁️",
    layout="centered"
)

# ======================================================
# PREMIUM AI + HEALTHCARE THEME (FIXED COLORS)
# ======================================================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0E1621, #06090F);
    font-family: 'Segoe UI', sans-serif;
}

/* ---------- TITLE ---------- */
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.4rem;
    color: #0077b6;
}

.sub-title {
    font-size: 1.25rem;
    text-align: center;
    color: #CFE9F7;
    margin-bottom: 2.6rem;
}

/* ---------- CARD ---------- */
.card {
    background: #FFFFFF;
    color: #1A1A1A;
    padding: 2.2rem;
    border-radius: 16px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.25);
    margin-top: 2rem;
}

/* ---------- RESULT STATES ---------- */
.result-good {
    border-left: 7px solid #2ECC71;
    background: #F0FDF4;
}

.result-warning {
    border-left: 7px solid #F39C12;
    background: #FFF7ED;
}

.result-danger {
    border-left: 7px solid #E74C3C;
    background: #FEF2F2;
}

/* ---------- HIGHLIGHT TEXT ---------- */
.highlight-label {
    font-size: 1.05rem;
    color: #1F2937;
    font-weight: 600;
    margin-top: 0.9rem;
}

.highlight-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #0077b6;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    font-size: 1rem;
    color: #A9B7C6;
    margin-top: 3rem;
    padding-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("final_best_model_EfficientNetB3.keras")
    with open("class_mapping.json") as f:
        data = json.load(f)
    return model, data["classes"], data["model_name"], data["test_accuracy"]

model, classes, model_name, accuracy = load_model_and_classes()

# ======================================================
# IMAGE PREPROCESS
# ======================================================
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img.astype("float32"))
    return np.expand_dims(img, axis=0)

# ======================================================
# PREDICTION
# ======================================================
def predict_disease(image):
    preds = model.predict(preprocess_image(image), verbose=0)[0]
    idx = np.argmax(preds)
    return classes[idx], preds[idx] * 100, preds * 100

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='main-title'>AI-Driven Retinal Disease Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>CNN-Based Transfer Learning with Medical Image Intelligence</div>", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.header("ℹ️ System Overview")
    st.markdown(f"""
**Model Architecture:** {model_name}  
**Test Accuracy:** {accuracy:.2f}%

**Disease Classes**
- Cataract  
- Diabetic Retinopathy  
- Glaucoma  
- Normal
""")

    st.header("⚠️ Medical Disclaimer")
    st.warning(
        "This AI system is intended strictly for research and educational purposes. "
        "It must not be used as a substitute for professional medical diagnosis."
    )

# ======================================================
# MAIN CONTENT
# ======================================================
uploaded_file = st.file_uploader("📤 Upload Retinal Fundus Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Retinal Fundus Image", use_column_width=True)

    if st.button("🔍 Analyze Image", use_container_width=True):
        with st.spinner("Analyzing retinal patterns using deep learning..."):
            disease, confidence, probabilities = predict_disease(image)

        if disease == "normal":
            style = "result-good"
            advice = "No significant retinal abnormalities detected. Routine eye check-ups are recommended."
        elif disease == "cataract":
            style = "result-warning"
            advice = "Cataract-related visual patterns detected. Early consultation with an ophthalmologist is advised."
        else:
            style = "result-danger"
            advice = "Patterns associated with serious retinal conditions detected. Immediate medical consultation is strongly recommended."

        # ======================================================
        # PREDICTION RESULT (FIXED – NO CODE BLOCK)
        # ======================================================
        st.markdown(f"""
<div class="card {style}">
<h2>🩺 Prediction Result</h2>

<p class="highlight-label">
Detected Condition:
<span class="highlight-value">{disease.replace('_',' ').title()}</span>
</p>

<p class="highlight-label">
Confidence Level:
<span class="highlight-value">{confidence:.2f}%</span>
</p>

<p style="margin-top:1.2rem; font-size:1.05rem; color:#374151;">
{advice}
</p>
</div>
""", unsafe_allow_html=True)

        # ======================================================
        # BAR CHART
        # ======================================================
        fig = go.Figure(go.Bar(
            x=[c.replace("_"," ").title() for c in classes],
            y=[float(p) for p in probabilities],
            marker_color="#0077b6"
        ))
        fig.update_layout(
            title="Disease Probability Distribution",
            xaxis_title="Disease Class",
            yaxis_title="Confidence (%)",
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ======================================================
        # ABOUT SECTION
        # ======================================================
        st.markdown("""
<div class="card">
<h3>🧠 About This AI System</h3>
<p>
This system uses deep convolutional neural networks with transfer learning to analyze retinal fundus images.
It learns disease-specific visual patterns and provides probabilistic predictions.
</p>
<p>
From a computer science perspective, this project demonstrates applied deep learning, medical image processing,
and end-to-end AI system deployment.
</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<div class="footer">
Developed by
<a href="https://masfiqur-nehal.vercel.app/" target="_blank"><b>Md. Masfiqur Rahman Nehal</b></a><br>
Department of Computer Science and Engineering<br>
Daffodil International University
</div>
""", unsafe_allow_html=True)
