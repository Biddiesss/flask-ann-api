from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("iris_ann_model.h5")

# Initialize Flask app
app = Flask("Iris API")

@app.route("/")
def home():
    return "API Iris est opérationnelle"

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]

    # Ensure the input is a numpy array with correct shape
    features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    return jsonify({"predicted_class": predicted_class})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
