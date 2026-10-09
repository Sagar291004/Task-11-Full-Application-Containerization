
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load Task 6 trained CNN model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "task6_cnn_model.keras")
)

model = tf.keras.models.load_model(MODEL_PATH)

# CIFAR-10 class names
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


@app.route("/")
def home():
    return "Flask API is running successfully!"


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if an image was uploaded
        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded. Please send an image file."
            }), 400

        file = request.files["image"]

        if not file.filename:
            return jsonify({
                "error": "No file selected."
            }), 400

        # Open and prepare image
        image = Image.open(file).convert("RGB")
        image = image.resize((32, 32))

        # Convert image to NumPy array
        image_array = np.array(image, dtype=np.float32)

        # Normalize pixel values
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        predictions = model.predict(image_array, verbose=0)

        # Get predicted class
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = class_names[predicted_index]

        # Calculate confidence percentage
        confidence = float(
            predictions[0][predicted_index] * 100
        )

        # Return prediction to Streamlit
        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        }), 200

    except Exception as e:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
