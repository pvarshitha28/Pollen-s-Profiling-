# app.py
import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from keras.preprocessing import image as keras_image_preprocessing # Alias to avoid conflict with PIL Image
import numpy as np

# --- Flask App Configuration ---
app = Flask(__name__)

# Define the folder to store uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Path to the trained model file
MODEL_PATH = 'pollen_cnn_model.h5'

# Global variable to hold the loaded model
model = None

# Global variable to hold class names.
# !!! IMPORTANT !!!
# After running train_model.py, copy the printed 'CLASS_NAMES' list
# from its console output and paste it here to replace this placeholder.
# Example: CLASS_NAMES = ['Ambrosia', 'Betula', 'Quercus', 'Pinus', 'Ulmus']
 # <<< REPLACE THIS EMPTY LIST WITH THE ACTUAL LIST FROM train_model.py's OUTPUT
CLASS_NAMES = [
    'anadenanthera',
    'arecaceae',
    'arrabidaea',
    'cecropia',
    'chromolaena',
    'combretum',
    'croton',
    'dipteryx',
    'eucalipto',
    'faramea',
    'hyptis',
    'mabea',
    'matayba',
    'mimosa',
    'myrcia',
    'protium',
    'qualea',
    'schinus',
    'senegalia',
    'serjania',
    'syagrus',
    'tridax',
    'urochloa'
]

def allowed_file(filename):
    """Checks if a file's extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_pollen_model():
    """
    Loads the pre-trained Keras model.
    Checks if the model file exists, otherwise prints an error.
    """
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Successfully loaded trained model from {MODEL_PATH}")
            # Verify that CLASS_NAMES has been updated by the user
            if not CLASS_NAMES:
                print("WARNING: CLASS_NAMES list in app.py is empty. Please update it with the classes from train_model.py output.")
            else:
                print(f"Model will predict among {len(CLASS_NAMES)} classes: {CLASS_NAMES}")
        except Exception as e:
            print(f"ERROR: Could not load model from {MODEL_PATH}. Reason: {e}")
            print("Please ensure your model is trained and saved correctly by running train_model.py first.")
            model = None # Set model to None to indicate failure
    else:
        print(f"ERROR: Model file '{MODEL_PATH}' not found.")
        print("Please run 'python train_model.py' first to train and save the model.")
        model = None

# Load the model when the Flask application starts
# This approach ensures the model is loaded once at app startup
with app.app_context():
    load_pollen_model()


# --- Prediction Function ---
def predict_pollen_type(image_path):
    """
    Preprocesses an image and makes a prediction using the loaded CNN model.
    Args:
        image_path (str): The path to the image file.
    Returns:
        tuple: A tuple containing the predicted class name and its probability.
    Raises:
        ValueError: If the model is not loaded or CLASS_NAMES is empty.
    """
    if model is None:
        raise ValueError("Model not loaded. Cannot perform prediction. Check server logs.")
    if not CLASS_NAMES:
        raise ValueError("CLASS_NAMES list is empty. Cannot map prediction to class name. Please update app.py.")

    # Load the image and resize it to the model's expected input shape
    # Ensure this target_size matches the input_shape used in train_model.py
    img = keras_image_preprocessing.load_img(image_path, target_size=(224, 224))
    img_array = keras_image_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, H, W, C)
    img_array /= 255.0 # Normalize pixel values to [0, 1]

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]

    if predicted_class_index < len(CLASS_NAMES):
        predicted_class_name = CLASS_NAMES[predicted_class_index]
    else:
        predicted_class_name = "Unknown" # Fallback if index is out of bounds
        print(f"Warning: Predicted class index {predicted_class_index} is out of bounds for CLASS_NAMES ({len(CLASS_NAMES)} classes).")

    return predicted_class_name, float(confidence)

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main page of the application."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles image uploads and performs pollen classification."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure it is trained and saved.'}), 503 # Service Unavailable

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Securely save the uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Perform prediction
            predicted_class, confidence = predict_pollen_type(filepath)
            return jsonify({
                'success': True,
                'filename': filename,
                'predicted_class': predicted_class,
                'confidence': f"{confidence*100:.2f}%"
            })
        except ValueError as ve:
            print(f"Prediction Error: {ve}")
            return jsonify({'error': str(ve)}), 500
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': f'Error processing image for prediction: {e}. Check server logs.'}), 500
        finally:
            # Clean up: remove the temporarily saved file
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)

