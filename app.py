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
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Google Drive model URL & local path
MODEL_URL = "https://drive.google.com/uc?export=download&id=1wBwrM4--8IeIcoyM9b_K8m2gUDEeQDvn"
MODEL_PATH = "butterfly_model_v1.h5"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True, quiet=False)
    print("✅ Download complete.")

# Load model
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Class labels
labels = ["ADONIS", "AFRICAN GIANT SWALLOWTAIL", "AMERICAN SNOOT", "AN 88", "APPOLLO", "ARCIGERA FLOWER MOTH", "ATALA",
          "ATLAS MOTH", "BANDED ORANGE HELICONIAN", "BANDED PEACOCK", "BANDED TIGER MOTH", "BECKERS WHITE", "BIRD CHERRY ERMINE MOTH",
          "BLACK HAIRSTREAK", "BLUE MORPHO", "BLUE SPOTTED CROW", "BROOKES BIRDWING", "BROWN ARGUS", "BROWN SIPROETA", "CABBAGE WHITE",
          "CAIRNS BIRDWING", "CHALK HILL BLUE", "CHECQUERED SKIPPER", "CHESTNUT", "CINNABAR MOTH", "CLEARWING MOTH", "CLEOPATRA",
          "CLODIUS PARNASSIAN", "CLOUDED SULPHUR", "COMET MOTH", "COMMON BANDED AWL", "COMMON WOOD-NYMPH", "COPPER TAIL", "CRECENT",
          "CRIMSON PATCH", "DANAID EGGFLY", "EASTERN COMA", "EASTERN DAPPLE WHITE", "EASTERN PINE ELFIN", "ELBOWED PIERROT",
          "EMPEROR GUM MOTH", "GARDEN TIGER MOTH", "GIANT LEOPARD MOTH", "GLITTERING SAPPHIRE", "GOLD BANDED", "GREAT EGGFLY",
          "GREAT JAY", "GREEN CELLED CATTLEHEART", "GREEN HAIRSTREAK", "GREY HAIRSTREAK", "HERCULES MOTH", "HUMMING BIRD HAWK MOTH",
          "INDRA SWALLOW", "IO MOTH", "Iphiclus sister", "JULIA", "LARGE MARBLE", "LUNA MOTH", "MADAGASCAN SUNSET MOTH", "MALACHITE",
          "MANGROVE SKIPPER", "MESTRA", "METALMARK", "MILBERTS TORTOISESHELL", "MOURNING CLOAK", "Monarch", "OLEANDER HAWK MOTH",
          "ORANGE OAKLEAF", "ORANGE TIP", "ORCHARD SWALLOW", "PAINTED LADY", "PAPER KITE", "PEACOCK", "PINE WHITE", "PIPEVINE SWALLOW",
          "POLYPHEMUS MOTH", "POPINJAY", "PURPLE HAIRSTREAK", "PURPLISH COPPER", "PaintedLady", "QUESTION MARK", "RED ADMIRAL",
          "RED CRACKER", "RED POSTMAN", "RED SPOTTED PURPLE", "ROSY MAPLE MOTH", "SCARCE SWALLOW", "SILVER SPOT SKIPPER",
          "SIXSPOT BURNET MOTH", "SLEEPY ORANGE", "SOOTYWING", "SOUTHERN DOGFACE", "STRAITED QUEEN", "Swallowtail", "TROPICAL LEAFWING",
          "TWO BARRED FLASHER", "ULYSES", "VICEROY", "WHITE LINED SPHINX MOTH", "WOOD SATYR", "YELLOW SWALLOW TAIL", "ZEBRA LONG WING"]

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

        # Preprocess image
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

