import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


app = Flask(__name__)

MODEL_NAME = "movenet_singlepose_lightning.tflite"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", MODEL_NAME)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = int(input_details[0]["shape"][1])


def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    scale = min(input_size / w, input_size / h)

    nw = int(w * scale)
    nh = int(h * scale)

    resized = cv2.resize(img, (nw, nh))

    padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    top = (input_size - nh) // 2
    left = (input_size - nw) // 2
    padded[top:top + nh, left:left + nw] = resized

    #return np.expand_dims(padded, axis=0).astype(np.uint8)
    return np.expand_dims(padded.astype(np.float32), axis=0)


@app.route("/infer", methods=["POST"])
def infer():
    data = request.json
    image_b64 = data["image"]

    image_bytes = base64.b64decode(image_b64)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    input_image = preprocess_frame(frame)

    interpreter.set_tensor(input_details[0]["index"], input_image)
    interpreter.invoke()

    keypoints = interpreter.get_tensor(output_details[0]["index"])

    return jsonify({
        "keypoints": keypoints.tolist()
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055)