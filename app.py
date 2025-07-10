import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Model download URL and path
MODEL_URL = "https://drive.google.com/uc?export=download&id=1wBwrM4--8IeIcoyM9b_K8m2gUDEeQDvn"
MODEL_PATH = "butterfly_model_v1.keras"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True, quiet=False)
    print("✅ Download complete.")

# Load trained model
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Load class labels (replace this with your full label list)
labels = [
    "ADONIS",
    "AFRICAN GIANT SWALLOWTAIL",
    "AMERICAN SNOOT",
    "AN 88",
    "APPOLLO"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict
        preds = model.predict(img_array)
        predicted_class = labels[np.argmax(preds)]

        return render_template('result.html', prediction=predicted_class, uploaded_filename=filename)
    return redirect('/')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
