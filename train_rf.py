import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load Dataset (Ensure nsl_kdd_dataset.csv is in the same folder)
df = pd.read_csv("nsl_kdd_dataset.csv")

# 2. Binary Labeling
df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

# 3. Handle Categorical Data
# We save the column names to ensure the app knows the order
categorical_cols = ["protocol_type", "service", "flag"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save for future use if needed

# 4. Split Features and Labels
X = df.drop("label", axis=1)
y = df["label"]
feature_names = X.columns.tolist()

# 5. Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 7. Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 8. Save Feature Importance for the UI
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)
importance_df.to_csv("feature_importance.csv", index=False)

# 9. Save Models and Metadata
joblib.dump(rf, "random_forest.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")

print("✅ Training complete. Files saved: random_forest.pkl, scaler.pkl, feature_importance.csv")