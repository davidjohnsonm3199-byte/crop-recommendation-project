🌾 Crop Recommendation System

A Machine Learning–powered application that suggests the most suitable crop for cultivation based on soil and climatic conditions.
This project includes a full Streamlit web app, preprocessing pipeline, and multiple trained ML models.

📌 Overview

The Crop Recommendation System predicts the ideal crop to cultivate using important features such as Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.
The project uses:

Imputer – handles missing values

Transformer – applies log transformation

Scaler – standardizes data

ML Models – Logistic Regression, Random Forest, and XGBoost

Streamlit app – for user-friendly predictions

All models and preprocessing tools are loaded from pickle files and run inside a simple, interactive UI.

🚀 Features

🌱 Predicts the best crop for given soil and climatic inputs

🤖 Supports three ML models

📊 End-to-end pipeline: imputation → log transform → scaling → prediction

🖼️ Displays matching crop image automatically

🧩 Modular and production-ready Streamlit script



🛠️ Technologies Used

Python

Streamlit

Pandas / NumPy

Scikit-Learn

XGBoost

Pickle
