🛡️ CyberShield AI – Intrusion Detection System
CyberShield AI is a Machine Learning-powered Network Intrusion Detection System (NIDS). It uses a Random Forest Classifier trained on the NSL-KDD dataset to analyze network traffic features and classify them as either Normal or an Attack (Anomalous).

The project includes a web-based dashboard built with Streamlit, providing real-time predictions and model interpretability via Feature Importance visualization.

🚀 Key Features
Multi-Model Foundation: Built to support Random Forest, Logistic Regression, and Artificial Neural Networks (ANN).

Instant Classification: Enter 41 network features to get an immediate security verdict.

Explainable AI (XAI): Visualizes the top 10 most influential network features using a dynamic bar chart.

Scalable Preprocessing: Includes a saved StandardScaler to ensure input data matches the training distribution.

📂 Project Structure
Plaintext

├── app.py                 # Streamlit Web Interface
├── train.py               # Training script for Random Forest & Scaler
├── nsl_kdd_dataset.csv    # The dataset (ensure this is in the root)
├── random_forest.pkl      # Saved Random Forest model
├── scaler.pkl             # Saved StandardScaler object
├── feature_names.pkl      # Saved list of feature names
└── feature_importance.csv # Data for the UI chart
🛠️ Installation & Setup
1. Clone the repository
Bash

git clone https://github.com/ShkS44D/AI-Project
cd cybershield-ai
2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:

Bash

pip install streamlit pandas numpy scikit-learn joblib
3. Train the Model
Before running the app, you must train the model to generate the necessary .pkl files:

Bash

python train.py
4. Launch the Dashboard
Bash

streamlit run app.py
📊 Dataset Insight
This project utilizes the NSL-KDD dataset, an improved version of the classic KDD'99. It consists of 41 features including:

Intrinsic features: Duration, protocol type, service, flag, etc.

Content features: Number of failed login attempts, logged-in status, etc.

Traffic features: Count, serror_rate, rerror_rate, etc.

🧩 How It Works
Data Preprocessing: The train.py script encodes categorical labels and scales numerical values using StandardScaler.

Model Training: A Random Forest model is trained to distinguish between legitimate traffic and various attack types (DoS, Probe, R2L, U2R).

Inference: The app.py script takes raw user input, applies the saved scaler, and produces a prediction with a confidence score.

Interpretability: The app displays which features (e.g., src_bytes, dst_host_srv_count) most heavily influenced the AI's decision.

⚠️ Disclaimer
This is a research prototype designed for educational purposes and is not intended to replace enterprise-grade firewall solutions. It is trained specifically on the NSL-KDD dataset characteristics.

Author: Saad Ahmed


License: MIT
