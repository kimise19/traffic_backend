from flask import request, Flask, jsonify
from PIL import Image
import base64
from io import BytesIO
from ultralytics import YOLO

app = Flask(__name__)

@app.route("/")
def root():
    with open("index.html") as file:
        return file.read()

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.json
        image_base64 = data.get("image")

        if not image_base64:
            raise ValueError("Image data is missing or empty.")

        # Decode base64 string to bytes
        image_bytes = base64.b64decode(image_base64)

        # Convert bytes to image using BytesIO
        image = Image.open(BytesIO(image_bytes))

        # Make sure the 'best.pt' file is in the correct directory or provide the full path
        model = YOLO("best.pt")

        results = model.predict(image)
        result = results[0]
        output = []

        for box in result.boxes:
            x1, y1, x2, y2 = [
                round(x) for x in box.xyxy[0].tolist()
            ]
            class_id = box.cls[0].item()
            prob = round(box.conf[0].item(), 2)
            output.append([
                x1, y1, x2, y2, result.names[class_id], prob
            ])

        return jsonify(output)

    except Exception as e:
        return jsonify([{"error": str(e)}])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
