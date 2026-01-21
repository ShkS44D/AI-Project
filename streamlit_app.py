import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Page Config
st.set_page_config(page_title="CyberShield AI", page_icon="🛡️")

st.title("🛡️ CyberShield AI – Intrusion Detection")
st.markdown("---")

# Load models and metadata safely
@st.cache_resource
def load_assets():
    rf = joblib.load("random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    feat_names = joblib.load("feature_names.pkl")
    return rf, scaler, feat_names

try:
    rf, scaler, feature_names = load_assets()
except Exception as e:
    st.error(f"Error loading model files. Did you run train.py? Details: {e}")
    st.stop()

# Layout: Two Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Network Data")
    features_input = st.text_area(
        "Paste 41 comma-separated features:",
        placeholder="0.0, 1.2, 0, 0.5...",
        help="Ensure you have exactly 41 numerical values."
    )
    
    predict_btn = st.button("Analyze Traffic", use_container_width=True)

with col2:
    st.subheader("Analysis Result")
    if predict_btn:
        try:
            # Convert input string to numpy array
            raw_data = [float(x.strip()) for x in features_input.split(",")]
            
            if len(raw_data) != 41:
                st.error(f"Input has {len(raw_data)} features. The model requires exactly 41.")
            else:
                # Processing
                data_array = np.array(raw_data).reshape(1, -1)
                data_scaled = scaler.transform(data_array)
                
                # Prediction
                prediction = rf.predict(data_scaled)
                probability = rf.predict_proba(data_scaled)[0][1]

                if prediction[0] == 1:
                    st.error(f"🚨 **ATTACK DETECTED**\n\nConfidence: {probability:.2%}")
                else:
                    st.success(f"✅ **NORMAL TRAFFIC**\n\nConfidence: {(1-probability):.2%}")
        
        except ValueError:
            st.error("Invalid Input: Please ensure all values are numbers.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Feature Importance Section
st.divider()
st.subheader("🔍 Model Insights: Top 10 Influence Factors")

if os.path.exists("feature_importance.csv"):
    importance_df = pd.read_csv("feature_importance.csv")
    top_10 = importance_df.head(10)
    st.bar_chart(data=top_10, x="Feature", y="Importance", color="#ff4b4b")
    st.caption("This chart shows which network features most heavily influence the AI's decision-making.")
else:
    st.info("Run the training script to generate feature importance data.")