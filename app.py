import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

import torch
from models.cnn import IntelCNN_PyTorch

# ── Config ────────────────────────────────────────────────────
CLASSES         = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE        = 150
PYTORCH_WEIGHTS = os.path.join(BASE_DIR, "geraud_model.pth")
KERAS_WEIGHTS   = os.path.join(BASE_DIR, "geraud_model.keras")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# ── Load PyTorch model ────────────────────────────────────────
pytorch_model = None
if os.path.exists(PYTORCH_WEIGHTS):
    pytorch_model = IntelCNN_PyTorch(num_classes=6).to(DEVICE)
    pytorch_model.load_state_dict(torch.load(PYTORCH_WEIGHTS, map_location=DEVICE))
    pytorch_model.eval()
    print(f"✅ PyTorch model loaded ({DEVICE})")
else:
    print(f"⚠️  Weights not found: {PYTORCH_WEIGHTS}")

# ── Load Keras model ──────────────────────────────────────────
keras_model = None
if os.path.exists(KERAS_WEIGHTS):
    import tensorflow as tf
    keras_model = tf.keras.models.load_model(KERAS_WEIGHTS)
    print("✅ Keras model loaded")
else:
    print(f"⚠️  Weights not found: {KERAS_WEIGHTS}")

# ── Preprocessing ─────────────────────────────────────────────
import torchvision.transforms as T

_torch_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_torch(pil_img):
    return _torch_tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)

def preprocess_keras(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    backend = request.form.get("backend", "PyTorch")
    file    = request.files["image"]
    img     = Image.open(io.BytesIO(file.read()))

    try:
        if backend == "PyTorch":
            if pytorch_model is None:
                return jsonify({"error": "PyTorch model not loaded"}), 500
            with torch.no_grad():
                logits = pytorch_model(preprocess_torch(img))
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        elif backend == "Keras":
            if keras_model is None:
                return jsonify({"error": "Keras model not loaded"}), 500
            probs = keras_model.predict(preprocess_keras(img), verbose=0)[0]

        else:
            return jsonify({"error": "Unknown backend"}), 400

        results = [
            {"class": cls, "confidence": round(float(p) * 100, 2)}
            for cls, p in zip(CLASSES, probs)
        ]
        results.sort(key=lambda x: x["confidence"], reverse=True)

        return jsonify({
            "prediction": results[0]["class"],
            "confidence": results[0]["confidence"],
            "all":        results,
            "backend":    backend,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/models", methods=["GET"])
def available_models():
    return jsonify({
        "PyTorch": pytorch_model is not None,
        "Keras":   keras_model   is not None,
    })

# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)