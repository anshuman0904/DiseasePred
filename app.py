from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS

class ModelHelper:
    def __init__(self, model_path, vectorizer_path, label_encoder_path):
        self.model = models.load_model(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.labels = joblib.load(label_encoder_path).classes_

    def vectorize_text(self, text):
        preprocessed_text = text.lower()
        return self.vectorizer.transform([preprocessed_text]).toarray().astype(np.float32)

def predict_disease(model_helper, symptoms):
    try:
        input_vector = model_helper.vectorize_text(symptoms)
        output = model_helper.model.predict(input_vector)
        max_index = int(np.argmax(output))
        max_value = float(output[0][max_index])

        probabilities = output[0].tolist()
        top_predictions = []

        for _ in range(3):
            max_idx = np.argmax(probabilities)
            confidence = probabilities[max_idx]

            if confidence > 0 and max_idx < len(model_helper.labels):
                top_predictions.append({
                    'disease': model_helper.labels[max_idx],
                    'confidence': confidence,
                })
                probabilities[max_idx] = -1

        return {
            'disease': model_helper.labels[max_index],
            'confidence': max_value,
            'topPredictions': top_predictions,
        }

    except Exception as e:
        print(f'Error during prediction: {e}')
        raise

# Initialize the ModelHelper
# Ensure these files are in the correct directory
model_helper = ModelHelper(
    model_path="text_model.h5",
    vectorizer_path="tfidf_vectorizer.pkl",
    label_encoder_path="label_encoder.pkl"
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', '')
    prediction_result = predict_disease(model_helper, symptoms)
    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(debug=True)











