import os
import requests
import pandas as pd
import streamlit as st
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Product Defect Classifier", page_icon="🔍")

st.title("🔍 Product Defect Classifier")
st.write("Upload an image to detect manufacturing defects in real-time.")

# --- SECTION 1: Inference & Prediction ---
model_choice = st.selectbox("Select Model Architecture", ["ResNet-18 (Transfer)", "Custom CNN"])
model_type_param = "resnet" if "ResNet" in model_choice else "cnn"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Analyzing image..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(f"{API_URL}?model_type={model_type_param}", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Analysis Complete!")
                    st.write(f"**Prediction:** {result['prediction'].upper()}")
                    st.write(f"**Confidence:** {result['confidence_percentage']}%")
                    st.write(f"**Model Used:** {result['model_used']}")
                else:
                    st.error("Error communicating with prediction server.")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

# --- SECTION 2: Week 3 Experiment Tracking ---
st.markdown("---")
st.subheader("📊 Experiment Tracking History")

csv_path = "logs/experiment_history.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No experiment logs found yet. Run 'python log_experiments.py' to generate.")