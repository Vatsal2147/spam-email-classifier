from pathlib import Path
import pickle
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"

app = Flask(__name__, static_folder=".", static_url_path="")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({
            "error": "Model not found. Run train_model.py first."
        }), 500

    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Please enter a message."}), 400

    prediction = int(model.predict([text])[0])

    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba([text])[0]) * 100)
    else:
        confidence = None

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 2) if confidence is not None else None
    })


if __name__ == "__main__":
    app.run(debug=True)
