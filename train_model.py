import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset 'Mental Health Dataset.csv'...")
df = pd.read_csv("Mental Health Dataset.csv")

# Prepare X (all 16 features) and y (target)
X = df.drop("Coping_Struggles", axis=1)
y = df["Coping_Struggles"]

# Convert all feature columns to string to ensure safe encoding
for col in X.columns:
    X[col] = X[col].astype(str)

print(f"Training OrdinalEncoder on {X.shape[1]} features...")
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = ordinal_encoder.fit_transform(X)

print("Training LabelEncoder on target 'Coping_Struggles'...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))

print("Training Random Forest Classifier on all 16 features...")
best_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
best_rf.fit(X_encoded, y_encoded)

print("Saving updated pkl files...")
joblib.dump(best_rf, "best_rf_model.pkl")
joblib.dump(ordinal_encoder, "ordinal_encoder.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nSuccess! All 3 files (best_rf_model.pkl, ordinal_encoder.pkl, label_encoder.pkl) have been updated for 16 features.")
