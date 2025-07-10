import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gdown
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_URL = "https://drive.google.com/uc?export=download&id=1wBwrM4--8IeIcoyM9b_K8m2gUDEeQDvn"
MODEL_PATH = "butterfly_model_v1.keras"
model = None

labels = [ "ADONIS", "AFRICAN GIANT SWALLOWTAIL", "AMERICAN SNOOT", "AN 88", "APPOLLO" ]

def download_and_load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True, quiet=False)
        print("✅ Download complete.")

    if model is None:
        print("⚙️ Loading model...")
        model = load_model(MODEL_PATH)
        print("✅ Model loaded!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    download_and_load_model()  # <-- Load/download here safely

    file = request.files.get('file')
    if not file or file.filename == '':
        return redirect(request.url)
    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    img = image.load_img(path, target_size=(224, 224))
    arr = image.img_to_array(img)[None, ...] / 255.0
    preds = model.predict(arr)
    cls = labels[np.argmax(preds)]
    return render_template('result.html', prediction=cls, uploaded_filename=filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
