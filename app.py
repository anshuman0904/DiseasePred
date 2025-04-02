from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch import nn
from torchvision import transforms
import io
import base64
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="best_vit_xception.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

diabetes_model = joblib.load('diabetes_rf_model.pkl')
lungModel = joblib.load('lungs_randomForest.pkl')

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

class ModelHelper:
    def __init__(self, model_path, vectorizer_path, label_encoder_path):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Load vectorizer vocabulary from JSON
        with open(vectorizer_path, 'r') as file:
            self.vectorizer_vocab = json.load(file)

        # Load label encoder from JSON
        with open(label_encoder_path, 'r') as file:
            self.labels = json.load(file)

        # Store input-output details for TFLite model
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

        # Prepare stop words
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = text.lower()
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stop_words and word not in string.punctuation]
        preprocessed_text = " ".join(filtered_words)
        return preprocessed_text

    def vectorize_text(self, text):
        preprocessed_text = self.preprocess_text(text)
        words = preprocessed_text.split()
        vector = np.zeros(len(self.vectorizer_vocab), dtype=np.float32)

        for word in words:
            if word in self.vectorizer_vocab:
                vector[self.vectorizer_vocab[word]] = 1.0  

        return vector.reshape(1, -1)

def predict_disease(model_helper, symptoms):
    try:
        input_vector = model_helper.vectorize_text(symptoms)
        input_tensor = tf.convert_to_tensor(input_vector, dtype=tf.float32)
        model_helper.interpreter.set_tensor(model_helper.input_index, input_tensor)
        model_helper.interpreter.invoke()
        output = model_helper.interpreter.get_tensor(model_helper.output_index)[0]

        max_index = int(np.argmax(output))
        max_value = float(output[max_index])

        probabilities = output.tolist()
        top_predictions = []

        for _ in range(3):
            max_idx = np.argmax(probabilities)
            confidence = probabilities[max_idx]

            if confidence > 0 and str(max_idx) in model_helper.labels:
                top_predictions.append({
                    'disease': model_helper.labels[str(max_idx)],
                    'confidence': confidence,
                })
                probabilities[max_idx] = -1

        return {
            'disease': model_helper.labels[str(max_index)],
            'confidence': max_value,
            'topPredictions': top_predictions,
        }
    except Exception as e:
        print(f'Error during prediction: {e}')
        raise

model_helper = ModelHelper(
    model_path="text_model.tflite",
    vectorizer_path="tfidf_vocab.json",
    label_encoder_path="labels.json"
)

# Prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', '')
    prediction_result = predict_disease(model_helper, symptoms)
    return jsonify(prediction_result)

class ImprovedTinyVGGModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super(ImprovedTinyVGGModel, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(hidden_units),
            nn.Dropout(0.2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, 4 * hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * hidden_units, 4 * hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(4 * hidden_units),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(4 * hidden_units, 4 * hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * hidden_units, 2 * hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(2 * hidden_units),
            nn.Dropout(0.2)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(2 * hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, output_shape, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(output_shape),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1176, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.classifier(x)
        return x

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = ImprovedTinyVGGModel(3, 48, 6).to(device)
loaded_model.load_state_dict(torch.load("MultipleEyeDiseaseDetectModel.pth", map_location=device))
loaded_model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match your model's expected input size
    transforms.ToTensor(),
])

# Function to predict eye disease
def predict_eye_disease(image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = loaded_model(image_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()

    class_names = ['AMD', 'Cataract', 'Glaucoma', 'Myopia', 'Non-Eye', 'Normal']
    predicted_class = class_names[np.argmax(probabilities)]

    top_predictions = []
    for i in np.argsort(probabilities)[-3:][::-1]:
        top_predictions.append({
            'disease': class_names[i],
            'confidence': float(probabilities[i])
        })

    return predicted_class, top_predictions

@app.route('/predict_eye', methods=['POST'])
def predict_eye():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    
    try:
        image = Image.open(image_file).convert("RGB")
        predicted_class, top_predictions = predict_eye_disease(image)
        
        return jsonify({
            'disease': predicted_class,
            'confidence': top_predictions[0]['confidence'],
            'topPredictions': top_predictions
        })
    
    except Exception as e:
        app.logger.error(f"Error in predict_eye: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Retrieve data from form
        data = [
            int(request.form['Age']),
            int(request.form['Gender']),  # 1 for Male, 0 for Female
            int(request.form['Polyuria']),
            int(request.form['Polydipsia']),
            int(request.form['Sudden_weight_loss']),
            int(request.form['Weakness']),
            int(request.form['Polyphagia']),
            int(request.form['Genital_thrush']),
            int(request.form['Visual_blurring']),
            int(request.form['Itching']),
            int(request.form['Irritability']),
            int(request.form['Delayed_healing']),
            int(request.form['Partial_paresis']),
            int(request.form['Muscle_stiffness']),
            int(request.form['Alopecia']),
            int(request.form['Obesity'])
        ]

        # Convert data to NumPy array
        custom_case = np.array([data])

        # Make prediction
        prediction = diabetes_model.predict(custom_case)
        prediction_label = 'Positive' if prediction[0] == 1 else 'Negative'
        print(prediction_label)

        # Return result
        return jsonify({"prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/diabetes', methods=['GET'])
def diabetes():
    return render_template('diabetes.html')

@app.route('/predict_lungs', methods=['POST'])
def predict_lungs():
    try:
        # Get the JSON data from the request
        data = request.json

        # Map keys to the names expected by the model
        renamed_data = {
            'gender': data['gender'],
            'age': data['age'],
            'smoking': data['smoking'],
            'yellow_fingers': data['yellow_fingers'],
            'anxiety': data['anxiety'],
            'peer_pressure': data['peer_pressure'],
            'chronic disease': data['chronic_disease'],
            'fatigue': data['fatigue'],
            'allergy': data['allergy'],
            'wheezing': data['wheezing'],
            'alcohol consuming': data['alcohol_consuming'],
            'coughing': data['coughing'],
            'shortness of breath': data['shortness_of_breath'],
            'swallowing difficulty': data['swallowing_difficulty'],
            'chest pain': data['chest_pain']
        }

        # Extract features from the renamed data
        features = [int(renamed_data[key]) for key in renamed_data]

        # Convert the features to a DataFrame
        columns = list(renamed_data.keys())
        input_data = pd.DataFrame([features], columns=columns)

        # Make prediction
        prediction = lungModel.predict(input_data)[0]
        prediction_label = 'YES' if prediction == 1 else 'NO'
        print(prediction_label)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction_label})

    except Exception as e:
        print("error: " + str(e))
        return jsonify({'error': str(e)}), 400

@app.route('/lungs', methods=['GET'])
def lungs():
    return render_template('lungs.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)