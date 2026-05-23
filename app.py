import streamlit as st
import pandas as pd
import joblib

# Load the model and encoders
best_rf = joblib.load("best_rf_model.pkl")
ordinal_encoder = joblib.load("ordinal_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load the dataset to get the column names for the selectbox options
df = pd.read_csv("Mental Health Dataset.csv")


# Prepare X (features) and Y (target) based on your dataset
X = df.drop("Coping_Struggles", axis=1)

# Streamlit app
st.title("🧠 Mental Health Coping Struggles Predictor")

st.write("### Enter behavioral responses to predict coping struggles:")

user_input = []
for col in X.columns:
    # Get unique values from the column for selectbox options
    options = df[col].unique().tolist()
    choice = st.selectbox(f"{col}", options)
    user_input.append(choice)

# Encode input and make prediction
user_input_encoded = ordinal_encoder.transform([user_input])
prediction = best_rf.predict(user_input_encoded)[0]
label = label_encoder.inverse_transform([prediction])[0]

# Display the prediction result
st.write("## Prediction:")
if label == "Yes":
    st.error("🚨 This individual may be struggling to cope.")
else:
    st.success("✅ This individual appears to be coping well.")
