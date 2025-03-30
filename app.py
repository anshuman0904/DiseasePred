from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import json
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

# Home Route - Renders the index.html page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', '')
    prediction_result = predict_disease(model_helper, symptoms)
    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import numpy as np
# import tensorflow as tf
# import json
# from sklearn.feature_extraction.text import TfidfVectorizer
# import nltk
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer

# # Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')

# app = Flask(__name__)
# CORS(app)  # Enable CORS

# class TFLiteModelHelper:
#     def __init__(self, model_path, vocab_path, labels_path):
#         # Load TFLite model
#         self.interpreter = tf.lite.Interpreter(model_path=model_path)
#         self.interpreter.allocate_tensors()
        
#         # Get input and output tensors
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()
        
#         # Load vocabulary and initialize TF-IDF vectorizer
#         with open(vocab_path, 'r') as f:
#             self.vocab_dict = json.load(f)
        
#         # Load labels
#         with open(labels_path, 'r') as f:
#             self.labels = json.load(f)
            
#         # Create and fit the vectorizer
#         self.vectorizer = self._create_fitted_vectorizer()
        
#         # Initialize stemmer and lemmatizer
#         self.stemmer = PorterStemmer()
#         self.lemmatizer = WordNetLemmatizer()
    
#     def _create_fitted_vectorizer(self):
#         """Create and fit a TfidfVectorizer with the loaded vocabulary"""
#         vectorizer = TfidfVectorizer(vocabulary=self.vocab_dict)
#         # Fit with a dummy document containing at least one term from vocabulary
#         dummy_text = " ".join(list(self.vocab_dict.keys())[:20])
#         vectorizer.fit([dummy_text])
#         return vectorizer
    
#     def preprocess_text(self, text, use_stemming=True, use_lemmatization=False):
#         """Preprocess text by applying stemming or lemmatization"""
#         # Tokenize the text
#         words = nltk.word_tokenize(text.lower())
        
#         processed_words = []
#         for word in words:
#             if use_stemming:
#                 # Apply stemming (faster but less accurate)
#                 processed_words.append(self.stemmer.stem(word))
#             elif use_lemmatization:
#                 # Apply lemmatization (slower but more accurate)
#                 processed_words.append(self.lemmatizer.lemmatize(word))
#             else:
#                 processed_words.append(word)
                
#         return " ".join(processed_words)

#     def vectorize_text(self, text, use_stemming=True, use_lemmatization=False):
#         """Convert text to TF-IDF vector with optional stemming/lemmatization"""
#         preprocessed_text = self.preprocess_text(text, use_stemming, use_lemmatization)
#         return self.vectorizer.transform([preprocessed_text]).toarray().astype(np.float32)
        
#     def predict(self, input_data):
#         """Run inference with TFLite model"""
#         # Set the input tensor
#         self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
#         # Run inference
#         self.interpreter.invoke()
        
#         # Get the output tensor
#         output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
#         return output_data

# def predict_disease(model_helper, symptoms, use_stemming=True, use_lemmatization=False):
#     try:
#         # Vectorize input text with NLP preprocessing
#         input_vector = model_helper.vectorize_text(symptoms, use_stemming, use_lemmatization)
        
#         # Run prediction with TFLite
#         output = model_helper.predict(input_vector)
        
#         # Process results
#         max_index = int(np.argmax(output[0]))
#         max_value = float(output[0][max_index])
        
#         probabilities = output[0].tolist()
#         top_predictions = []
        
#         for _ in range(3):
#             max_idx = np.argmax(probabilities)
#             confidence = probabilities[max_idx]
            
#             if confidence > 0 and max_idx < len(model_helper.labels):
#                 top_predictions.append({
#                     'disease': model_helper.labels[max_idx],
#                     'confidence': confidence,
#                 })
#             probabilities[max_idx] = -1
            
#         return {
#             'disease': model_helper.labels[max_index],
#             'confidence': max_value,
#             'topPredictions': top_predictions,
#         }
        
#     except Exception as e:
#         print(f'Error during prediction: {e}')
#         raise

# # Initialize the TFLite ModelHelper
# model_helper = TFLiteModelHelper(
#     model_path="text_model.tflite",
#     vocab_path="tfidf_vocab.json",
#     labels_path="labels.json"
# )

# # Home Route - Renders the index.html page
# @app.route('/', methods=['GET'])
# def home():
#     return render_template('index.html')

# # Prediction Route
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     symptoms = data.get('symptoms', '')
    
#     # Get NLP processing options (defaults to stemming)
#     use_stemming = data.get('use_stemming', True)
#     use_lemmatization = data.get('use_lemmatization', False)
    
#     prediction_result = predict_disease(model_helper, symptoms, use_stemming, use_lemmatization)
#     return jsonify(prediction_result)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
