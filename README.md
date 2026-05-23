# 🧠 Mental Health Coping Struggles Predictor

An interactive, user-friendly **Streamlit** web application that leverages a **Random Forest Classifier** machine learning model to predict whether an individual might be struggling to cope with mental health issues based on their behavioral and demographic attributes.

---

## 🚀 Features

- **Web-Based Interface**: Built with **Streamlit** for a modern, responsive, and interactive user experience.
- **Machine Learning Backend**: Powered by a **Random Forest Classifier** trained on key behavioral features.
- **16-Feature Predictor**: Utilizes detailed indicators like daily routine, mood fluctuations, self-employment status, social weakness, work interest, and more to provide predictions.
- **Self-Contained Training**: Includes a training script (`train_model.py`) to easily retrain the model and update encoders.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-Learn
- **Data Manipulation**: Pandas & NumPy
- **Model Persistence**: Joblib

---

## 📦 Installation & Setup

Follow these steps to set up and run the project locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/Ambika-7/mental-health-prediction-ml.git
cd mental-health-prediction-ml
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed, then install the required libraries:
```bash
pip install streamlit pandas joblib scikit-learn lightgbm
```

---

## 🏃 How to Run the Application

Start the Streamlit development server by running:

```bash
python -m streamlit run app.py
```

Once running, the application will automatically open in your default browser at:
👉 **`http://localhost:8501`**

---

## 🔄 How to Retrain the Model

If you wish to retrain the Random Forest model on the full 16-feature dataset, simply execute:

```bash
python train_model.py
```

This script will read `Mental Health Dataset.csv`, process the categorical features using `OrdinalEncoder`, train a new model, and save the updated pickle files (`best_rf_model.pkl`, `ordinal_encoder.pkl`, `label_encoder.pkl`) automatically.

---

## 📂 Project Structure

```text
├── Mental Health Dataset.csv  # The source dataset
├── app.py                     # Streamlit application code
├── train_model.py             # Model training script
├── best_rf_model.pkl          # Trained Random Forest model
├── ordinal_encoder.pkl        # Fitted feature encoder
├── label_encoder.pkl          # Fitted target label encoder
└── README.md                  # Project documentation
```

---

## 👥 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.
