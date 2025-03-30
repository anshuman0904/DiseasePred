from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import json
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)  # Enable CORS

class TFLiteModelHelper:
    def __init__(self, model_path, vocab_path, labels_path):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load vocabulary and initialize TF-IDF vectorizer
        with open(vocab_path, 'r') as f:
            self.vocab_dict = json.load(f)
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
            
        # Create and fit the vectorizer
        self.vectorizer = self._create_fitted_vectorizer()
    
    def _create_fitted_vectorizer(self):
        """Create and fit a TfidfVectorizer with the loaded vocabulary"""
        vectorizer = TfidfVectorizer(vocabulary=self.vocab_dict)
        # Fit with a dummy document containing at least one term from vocabulary
        # This is just to mark the vectorizer as fitted
        dummy_text = " ".join(list(self.vocab_dict.keys())[:20])  # Use first 20 words from vocab
        vectorizer.fit([dummy_text])
        return vectorizer

    def vectorize_text(self, text):
        """Convert text to TF-IDF vector"""
        preprocessed_text = text.lower()
        return self.vectorizer.transform([preprocessed_text]).toarray().astype(np.float32)
        
    def predict(self, input_data):
        """Run inference with TFLite model"""
        # Set the input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get the output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

def predict_disease(model_helper, symptoms):
    try:
        # Vectorize input text
        input_vector = model_helper.vectorize_text(symptoms)
        
        # Run prediction with TFLite
        output = model_helper.predict(input_vector)
        
        # Process results
        max_index = int(np.argmax(output[0]))
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

# Initialize the TFLite ModelHelper
model_helper = TFLiteModelHelper(
    model_path="text_model.tflite",
    vocab_path="tfidf_vocab.json",
    labels_path="labels.json"
)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="best_vit_xception.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mapping Indices to Classes
classes = [
    "Actinic Keratoses and Intraepithelial Carcinoma / Bowen's Disease",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi (Common Mole)",
    "Vascular Lesions"
]

def preprocess_image(image_file):
    img = Image.open(image_file).resize((72, 72))
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# Home Route - Renders the new home page
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# Symptom-based prediction route
@app.route('/symptom_prediction', methods=['GET'])
def symptom_prediction():
    return render_template('symptom_prediction.html')

# Skin disease classification route
@app.route('/skin_disease', methods=['GET'])
def skin_disease():
    return render_template('skin_disease.html')

@app.route('/predict_skin', methods=['POST'])
def predict_skin():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    
    try:
        input_data = preprocess_image(image_file)
        
        # Make predictions
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(output_data)
        predicted_confidence = float(output_data[0][predicted_index])
        
        predicted_class = classes[predicted_index]
        
        # Get top 3 predictions
        top_indices = np.argsort(output_data[0])[-3:][::-1]
        top_predictions = [
            {'disease': classes[i], 'confidence': float(output_data[0][i])}
            for i in top_indices
        ]
        
        return jsonify({
            'disease': predicted_class,
            'confidence': predicted_confidence,
            'topPredictions': top_predictions
        })
    
    except Exception as e:
        app.logger.error(f"Error in predict_skin: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Eye disease classification route
@app.route('/eye_disease', methods=['GET'])
def eye_disease():
    return render_template('eye_disease.html')

# Prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', '')
    prediction_result = predict_disease(model_helper, symptoms)
    return jsonify(prediction_result)

# Dummy API for eye disease prediction
@app.route('/predict_eye', methods=['POST'])
def predict_eye():
    # Dummy response for now
    return jsonify({
        'disease': 'Cataract',
        'confidence': 0.92,
        'topPredictions': [
            {'disease': 'Cataract', 'confidence': 0.92},
            {'disease': 'Glaucoma', 'confidence': 0.06},
            {'disease': 'Diabetic Retinopathy', 'confidence': 0.02}
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
