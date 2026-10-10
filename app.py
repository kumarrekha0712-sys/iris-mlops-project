
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.title("Iris Flower Classification")
st.write("Predict the species using a trained neural network.")

model = load_model("iris_model.keras")
scaler = joblib.load("scaler.pkl")

classes = ["setosa", "versicolor", "virginica"]

sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width", 0.0, 10.0, 0.2)

if st.button("Predict Species"):
    flower = np.array([[
        sepal_length, sepal_width,
        petal_length, petal_width
    ]])

    flower_scaled = scaler.transform(flower)
    probabilities = model.predict(flower_scaled, verbose=0)
    index = np.argmax(probabilities, axis=1)[0]

    st.success(f"Predicted species: {classes[index]}")
    st.write(f"Model confidence: {probabilities[0][index] * 100:.2f}%")
