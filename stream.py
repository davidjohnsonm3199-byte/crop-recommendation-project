import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

@st.cache_resource
def load_models_and_tools():
    
    with open(r"model_pickle_dump\imputer.pkl", "rb") as f:
        imputer = pickle.load(f)
    with open(r"model_pickle_dump\transformer.pkl", "rb") as f:
        transformer = pickle.load(f)
    with open(r"model_pickle_dump\scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    models = {}
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        with open(f"model_pickle_dump\{name}_model.pkl", "rb") as f:
            models[name] = pickle.load(f)
    return imputer, transformer, scaler, models

imputer, transformer, scaler, models = load_models_and_tools()

crop_labels = [
    'banana', 'blackgram', 'coconut', 'coffee', 'cotton', 'jute',
    'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon',
    'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
]

# Make sure you have these images in a folder: /crop_images/
# Filenames should match crops exactly (e.g., banana.jpg, coffee.jpg, etc.)
crop_images = [f"crop_images/{crop}.jpeg" for crop in crop_labels]


# Streamlit UI
st.title("🌾 Crop Recommendation App")
st.write("Provide soil and climate details to predict the most suitable crop for cultivation.")

feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
user_input = []

cols = st.columns(2)
for idx, feature in enumerate(feature_names):
    val = cols[idx % 2].number_input(f"{feature}:", step=1.0, format="%.2f")
    user_input.append(val)

input_df = pd.DataFrame([user_input], columns=feature_names)

model_choice = st.selectbox("Choose Model:", ["Logistic Regression", "Random Forest", "XGBoost"])
model_key = model_choice.lower().replace(" ", "_")
selected_model = models[model_key]


# Prediction Logic

if st.button("🔮 Predict Crop"):
    # 1️⃣ Handle missing values
    X_imputed = imputer.transform(input_df)
    
    # 2️⃣ Apply log transform
    X_log = transformer.transform(X_imputed)
    
    # 3️⃣ Scale data
    X_scaled = scaler.transform(X_log)
    
    # 4️⃣ Predict
    y_pred = selected_model.predict(X_scaled)[0]
    predicted_crop = crop_labels[int(y_pred)]
    
    st.success(f"### ✅ Recommended Crop: {predicted_crop.capitalize()}")
    st.write(f"**Model Used:** {model_choice}")
    
    # 5️⃣ Display corresponding image
    image_path = crop_images[int(y_pred)]
    if os.path.exists(image_path):
        st.image(image_path, caption=predicted_crop.capitalize())
    else:
        st.warning("⚠️ Image not found. Please add the crop image to the 'crop_images' folder.")

