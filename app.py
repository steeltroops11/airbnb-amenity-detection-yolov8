from flask import Flask, request, jsonify, render_template, send_file
from detector import AmenityDetector
import os
import base64

app = Flask(__name__)

# Get absolute path to model file to ensure it's always found
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "airbnb_amenity_yolov8_final.pt")

# Verify model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

detector = AmenityDetector(MODEL_PATH)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    # Generate unique filename to avoid conflicts
    import uuid
    unique_filename = f"{uuid.uuid4()}_{image.filename}"
    image_path = os.path.join(UPLOAD_DIR, unique_filename)
    image.save(image_path)

    # Detect with bounding boxes drawn
    detections, annotated_image_path = detector.detect(image_path, draw_boxes=True)

    # Convert annotated image to base64 for frontend display
    annotated_image_base64 = None
    if annotated_image_path and os.path.exists(annotated_image_path):
        with open(annotated_image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            annotated_image_base64 = f"data:image/jpeg;base64,{img_data}"

    return jsonify({
        "amenities_detected": detections,
        "count": len(detections),
        "annotated_image": annotated_image_base64
    })

if __name__ == "__main__":
    app.run(debug=True)
