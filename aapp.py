from flask import Flask, request, render_template, url_for
from tensorflow.keras.preprocessing import image, ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load model
model = load_model('butterfly_model_v1.h5')
print("✅ Model loaded successfully.")

# Prepare class labels from training directory
datagen = ImageDataGenerator(rescale=1./255)
train_flow = datagen.flow_from_directory(
    "dataset/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
labels = list(train_flow.class_indices.keys())
print("✅ Labels loaded:", labels)

# Home route — upload page
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "⚠️ No file uploaded", 400

    file = request.files['file']

    if file.filename == '':
        return "⚠️ No file selected", 400

    # Save file
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    # Prepare image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    preds = model.predict(img_array)
    predicted_species = labels[np.argmax(preds)]
    confidence = round(np.max(preds) * 100, 2)

    # Return result page
    return render_template('result.html',
                           predicted_species=predicted_species,
                           confidence=confidence,
                           uploaded_filename=file.filename)

# Run app on a free port (change here if needed)
if __name__ == '__main__':
    app.run(debug=False, port=5001)

