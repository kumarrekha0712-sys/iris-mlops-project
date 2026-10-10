
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the saved model and scaler
model = load_model("iris_model.keras")
scaler = joblib.load("scaler.pkl")

classes = ["setosa", "versicolor", "virginica"]

# Enter flower measurements
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

# Prepare and scale input
flower = np.array([[
    sepal_length, sepal_width,
    petal_length, petal_width
]])

flower_scaled = scaler.transform(flower)

# Predict flower species
prediction = model.predict(flower_scaled, verbose=0)
class_index = np.argmax(prediction, axis=1)[0]

print("Predicted flower:", classes[class_index])
