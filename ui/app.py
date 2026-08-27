import os
import requests
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Product Defect Classifier",
    page_icon="🔍",
    layout="wide"
)

# ==========================================
# 2. PATHS & LOG INITIALIZATION (OPTION A)
# ==========================================
API_URL = "http://127.0.0.1:8000/predict"
LOG_FILE_PATH = os.path.join("logs", "experiment_history.csv")
EXACT_COLUMNS = ["File Name", "Model Used", "Prediction", "Confidence (in %)"]

# Ensure logs directory exists and create a fresh CSV with exact requested columns
os.makedirs("logs", exist_ok=True)
if not os.path.exists(LOG_FILE_PATH):
    empty_df = pd.DataFrame(columns=EXACT_COLUMNS)
    empty_df.to_csv(LOG_FILE_PATH, index=False)

# ==========================================
# 3. DASHBOARD HEADER
# ==========================================
st.title("🔍 Product Defect Classification System")
st.markdown("Automated quality control inspection dashboard with live log monitoring.")

# ==========================================
# 4. SECTION 1: LIVE INFERENCE
# ==========================================
st.subheader("🎯 Real-Time Quality Inspection")
col1, col2 = st.columns([1, 1])

with col1:
    model_choice = st.selectbox(
        "Select Model Architecture",
        options=["resnet", "cnn"],
        format_func=lambda x: "ResNet" if x == "resnet" else "CNN"
    )
    uploaded_file = st.file_uploader("Choose a product photo...", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image Preview", use_container_width=True)

with col2:
    if uploaded_file is not None:
        if st.button("Run Inspection", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {"model_type": model_choice}
                    
                    headers = {'Connection': 'close'}
                    response = requests.post(API_URL, files=files, params=params, headers=headers, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        raw_result = str(data.get("prediction", "")).upper()
                        
                        # Standardize display result to OK or Defective
                        prediction_result = "OK" if raw_result == "OK" else "Defective"
                        confidence = data.get("confidence_percentage", 0.0)
                        model_used = "ResNet" if model_choice == "resnet" else "CNN"
                        filename = uploaded_file.name

                        if prediction_result == "OK":
                            st.success(f"**Status:** {prediction_result} — Passed Quality Control")
                        else:
                            st.error(f"**Status:** {prediction_result} — Defect Detected")
                        
                        st.metric("Confidence Score", f"{confidence}%")
                        st.info(f"**Active Model:** {model_used}")

                        # Save log entry with exact column schema
                        new_log = pd.DataFrame([{
                            "File Name": filename,
                            "Model Used": model_used,
                            "Prediction": prediction_result,
                            "Confidence (in %)": f"{confidence}%"
                        }])
                        new_log.to_csv(LOG_FILE_PATH, mode='a', header=False, index=False)
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Please restart your FastAPI backend server.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Verify `uvicorn serving.api:app --reload` is running on port 8000.")

st.divider()

# ==========================================
# 5. SECTION 2: MODEL BENCHMARK MATRIX
# ==========================================
st.subheader("📊 Model Architecture Benchmark")
st.caption("Side-by-side performance comparison across training runs.")

comparison_data = {
    "Model Name": ["Custom CNN Baseline", "ResNet-18 (Transfer Learning)"],
    "Epochs": [5, 5],
    "Batch Size": [32, 32],
    "Learning Rate": [0.0001, 0.0001],
    "Accuracy (%)": [97.37, 99.47],
    "Validation Loss": [0.0812, 0.0145]
}

st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

st.divider()

# ==========================================
# 6. SECTION 3: LOG MONITORING (SEPARATE)
# ==========================================
st.subheader("📋 Log Monitoring")

if os.path.exists(LOG_FILE_PATH):
    logs_df = pd.read_csv(LOG_FILE_PATH)

    if not logs_df.empty:
        log_col1, log_col2 = st.columns([3, 1])
        with log_col1:
            st.caption(f"Total Inspection Records: **{len(logs_df)}**")
        with log_col2:
            st.download_button(
                label="📥 Download Log (CSV)",
                data=logs_df.to_csv(index=False).encode('utf-8'),
                file_name="inspection_logs.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.dataframe(logs_df.sort_index(ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No inspection logs recorded yet. Run an inspection above to log data here.")
