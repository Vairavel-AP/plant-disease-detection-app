from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os
import logging
import google.generativeai as genai
import json

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Folder to temporarily store uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# File to cache precautions
PRECAUTIONS_FILE = "precautions.json"
if os.path.exists(PRECAUTIONS_FILE):
    with open(PRECAUTIONS_FILE, "r") as f:
        PRECAUTIONS_CACHE = json.load(f)
else:
    PRECAUTIONS_CACHE = {}

# Load the pre-trained model
try:
    model = tf.keras.models.load_model('trained_plant_disease_model.keras')
    model.make_predict_function()
    app.logger.info("✅ Model loaded successfully.")
except Exception as e:
    app.logger.error(f"❌ Error loading model: {e}")
    model = None

# Class labels (should match training dataset)
CLASS_NAMES = [
    'Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Blueberry_healthy', 'Cherry_(including_sour)_Powdery_mildew', 'Cherry_(including_sour)_healthy',
    'Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)_Common_rust_',
    'Corn_(maize)_Northern_Leaf_Blight', 'Corn_(maize)_healthy',
    'Grape_Black_rot', 'Grape_Esca_(Black_Measles)', 'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape_healthy',
    'Orange_Haunglongbing_(Citrus_greening)', 'Peach_Bacterial_spot', 'Peach_healthy',
    'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy',
    'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy',
    'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew',
    'Strawberry_Leaf_scorch', 'Strawberry_healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two-spotted_spider_mite',
    'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus', 'Tomato_healthy'
]

# Setup Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
ai_model = genai.GenerativeModel("gemini-1.5-flash")

# Image preprocessing
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    return input_arr

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not available'}), 500
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found in the request'}), 400
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        preprocessed_image = preprocess_image(image_path)
        predictions = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = CLASS_NAMES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        os.remove(image_path)  # cleanup

        return jsonify({
            'predicted_class': predicted_class_label,
            'confidence': confidence
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': 'Prediction failed'}), 500

# Gemini AI-generated precautions with caching
@app.route('/get_precautions', methods=['POST'])
def get_precautions():
    try:
        data = request.get_json()
        if not data or 'predicted_class' not in data:
            return jsonify({'error': 'No predicted class provided'}), 400

        predicted_class = data['predicted_class']

        # ✅ Check cache first
        if predicted_class in PRECAUTIONS_CACHE:
            return jsonify(PRECAUTIONS_CACHE[predicted_class])

        # If not cached, fetch from Gemini
        prompts = {
            'Preventive Measures': f"What preventive measures should be taken for {predicted_class} in 5 lines?",
            'Immediate Actions': f"What immediate actions should be taken for {predicted_class} in 5 lines?",
            'Long-Term Solutions': f"What are the long-term solutions for {predicted_class} in 5 lines?"
        }

        responses = {}
        for key, prompt in prompts.items():
            ai_response = ai_model.generate_content(prompt)
            if hasattr(ai_response, 'text') and ai_response.text.strip():
                responses[key] = ai_response.text.strip()
            else:
                responses[key] = "No response generated."

        # ✅ Save to cache
        PRECAUTIONS_CACHE[predicted_class] = responses
        with open(PRECAUTIONS_FILE, "w") as f:
            json.dump(PRECAUTIONS_CACHE, f, indent=4)

        return jsonify(responses)

    except Exception as e:
        app.logger.error(f"Gemini error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to generate precautions'}), 500

# HTML UI route
@app.route('/')
def home():
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
