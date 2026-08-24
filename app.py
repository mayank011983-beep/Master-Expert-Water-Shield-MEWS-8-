import io

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def decode_image(upload, field_name):
    if upload is None or not upload.filename:
        raise ValueError(f"Missing required '{field_name}' file")
    data = upload.read()
    if not data:
        raise ValueError(f"The '{field_name}' file is empty")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"The '{field_name}' file is not a valid image")
    return image


def split_target(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if image.shape[2] == 3:
        return image.copy(), None
    if image.shape[2] == 4:
        return image[:, :, :3].copy(), image[:, :, 3:4].copy()
    raise ValueError("The image has an unsupported color format")


def split_watermark(image):
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        alpha = np.full(image.shape, 255, dtype=np.uint8)
        return bgr, alpha
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        return image, alpha
    if image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3]
    raise ValueError("The watermark has an unsupported color format")


def apply_visible_shield(target, watermark):
    target_height, target_width = target.shape[:2]
    watermark_bgr, watermark_alpha = split_watermark(watermark)
    watermark_height, watermark_width = watermark_bgr.shape[:2]
    if target_width < 2 or target_height < 2 or watermark_width < 1 or watermark_height < 1:
        raise ValueError("The image dimensions are too small to apply a watermark")
    desired_width = max(1, int(round(target_width * 0.20)))
    desired_height = max(1, int(round(watermark_height * desired_width / watermark_width)))
    available_height = max(1, target_height - 2)
    if desired_height > available_height:
        scale = available_height / desired_height
        desired_width = max(1, int(round(desired_width * scale)))
        desired_height = available_height
    watermark_bgr = cv2.resize(watermark_bgr, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    watermark_alpha = cv2.resize(watermark_alpha, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    margin = max(0, min(16, target_width // 50, target_height // 50))
    x = max(0, target_width - desired_width - margin)
    y = max(0, target_height - desired_height - margin)
    region_width = min(desired_width, target_width - x)
    region_height = min(desired_height, target_height - y)
    watermark_bgr = watermark_bgr[:region_height, :region_width]
    alpha = watermark_alpha[:region_height, :region_width].astype(np.float32)[:, :, np.newaxis] / 255.0
    destination = target[y:y + region_height, x:x + region_width].astype(np.float32)
    target[y:y + region_height, x:x + region_width] = np.clip(
        destination * (1.0 - alpha) + watermark_bgr.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    return target


def apply_invisible_blue_noise(image):
    noise = np.random.normal(0.0, 0.75, image.shape[:2]).astype(np.float32)
    blue = image[:, :, 0].astype(np.float32)
    image[:, :, 0] = np.clip(blue + noise, 0, 255).astype(np.uint8)
    return image


@app.route("/", methods=["GET"])
def health_check():
    return "MEWS 8 Engine Online", 200


@app.route("/process", methods=["POST"])
def process_image():
    try:
        source = decode_image(request.files.get("image"), "image")
        watermark = decode_image(request.files.get("watermark"), "watermark")
        target, target_alpha = split_target(source)
        protected = apply_visible_shield(target, watermark)
        protected = apply_invisible_blue_noise(protected)
        if target_alpha is not None:
            protected = np.concatenate((protected, target_alpha), axis=2)
        success, encoded = cv2.imencode(".png", protected)
        if not success:
            raise RuntimeError("Failed to encode the protected PNG")
        return send_file(
            io.BytesIO(encoded.tobytes()),
            mimetype="image/png",
            as_attachment=True,
            download_name="mews8-protected.png",
        )
    except Exception as error:
        print(f"MEWS 8 processing error: {error}", flush=True)
        app.logger.exception("MEWS 8 processing error")
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
