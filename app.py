import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timezone
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image, PngImagePlugin
from werkzeug.exceptions import RequestEntityTooLarge


MAX_PIXELS = int(os.environ.get("MEWS8_MAX_PIXELS", "40000000"))
MAX_CONTENT_LENGTH = int(os.environ.get("MEWS8_MAX_UPLOAD_BYTES", str(30 * 1024 * 1024)))
BLOCK_SIZE = 8
QIM_STEP = 28.0
MESH_QIM_STEP = 12.0
WATERMARK_VERSION = "MEWS8/1"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


class ProcessingError(Exception):
    pass


def signing_key():
    value = os.environ.get("MEWS8_SIGNING_KEY")
    if not value:
        raise ProcessingError("MEWS8_SIGNING_KEY must be configured before images can be sealed")
    return value.encode("utf-8")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest(manifest):
    return hmac.new(signing_key(), canonical_json(manifest), hashlib.sha256).hexdigest()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def decode_upload(upload, label):
    if upload is None or not upload.filename:
        raise ProcessingError(f"missing {label} file")
    data = upload.read()
    if not data:
        raise ProcessingError(f"{label} file is empty")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ProcessingError(f"{label} file is not a supported image")
    if image.ndim < 2 or image.shape[0] < 8 or image.shape[1] < 8:
        raise ProcessingError(f"{label} image is too small")
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise ProcessingError(f"{label} image exceeds the pixel safety limit")
    return image


def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    raise ProcessingError("image has an unsupported channel layout")


def overlay_visible_mark(image, watermark):
    source = image.copy()
    mark = watermark
    image_height, image_width = source.shape[:2]
    mark_height, mark_width = mark.shape[:2]
    limit_width = max(1, int(image_width * 0.28))
    limit_height = max(1, int(image_height * 0.20))
    scale = min(limit_width / mark_width, limit_height / mark_height, 1.0)
    if scale < 1.0:
        mark = cv2.resize(mark, (max(1, round(mark_width * scale)), max(1, round(mark_height * scale))), interpolation=cv2.INTER_AREA)
    mark_height, mark_width = mark.shape[:2]
    margin = max(4, min(20, image_width // 30, image_height // 30))
    x = max(0, image_width - mark_width - margin)
    y = max(0, image_height - mark_height - margin)
    mark_width = min(mark_width, image_width - x)
    mark_height = min(mark_height, image_height - y)
    mark = mark[:mark_height, :mark_width]
    if mark.ndim == 2:
        mark_bgr = cv2.cvtColor(mark, cv2.COLOR_GRAY2BGR)
        alpha = np.full((mark_height, mark_width, 1), 0.30, dtype=np.float32)
    elif mark.shape[2] == 4:
        mark_bgr = mark[:, :, :3]
        alpha = mark[:, :, 3:4].astype(np.float32) / 255.0 * 0.30
    else:
        mark_bgr = mark[:, :, :3]
        alpha = np.full((mark_height, mark_width, 1), 0.30, dtype=np.float32)
    region = source[y:y + mark_height, x:x + mark_width].astype(np.float32)
    source[y:y + mark_height, x:x + mark_width] = np.clip(region * (1.0 - alpha) + mark_bgr.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return source


def bits_from_bytes(value):
    return np.unpackbits(np.frombuffer(value, dtype=np.uint8)).astype(np.uint8)


def payload_bits(serial, source_digest):
    seed = f"{WATERMARK_VERSION}|{serial}|{source_digest}".encode("ascii")
    return bits_from_bytes(hmac.new(signing_key(), seed, hashlib.sha256).digest())


def block_index(row, column, columns, count):
    return (row * columns + column) % count


def force_qim(coefficient, bit, strength=1.0):
    sign = -1.0 if coefficient < 0 else 1.0
    level = max(1, int(round(abs(float(coefficient)) / QIM_STEP)))
    if level % 2 != int(bit):
        level += 1
    return sign * level * QIM_STEP * strength


def force_mesh_qim(coefficient, bit):
    sign = -1.0 if coefficient < 0 else 1.0
    level = max(1, int(round(abs(float(coefficient)) / MESH_QIM_STEP)))
    if level % 2 != int(bit):
        level += 1
    return sign * level * MESH_QIM_STEP


def geometric_mesh(serial, source_digest, rows, columns):
    key = signing_key()
    previous = bytes(32)
    bits = []
    prefix = f"{WATERMARK_VERSION}|mesh|{serial}|{source_digest}|".encode("ascii")
    for index in range(rows * columns):
        previous = hmac.new(key, prefix + index.to_bytes(4, "big") + previous, hashlib.sha256).digest()
        bits.append(previous[0] & 1)
    return np.array(bits, dtype=np.uint8), previous.hex()


def embed_hidden_seal(image, serial, source_digest):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[:, :, 0].astype(np.float32)
    rows = luminance.shape[0] // BLOCK_SIZE
    columns = luminance.shape[1] // BLOCK_SIZE
    if rows * columns < 16:
        raise ProcessingError("image is too small for the MEWS 8 frequency seal")
    bits = payload_bits(serial, source_digest)
    mesh, mesh_root = geometric_mesh(serial, source_digest, rows, columns)
    rng_seed = int.from_bytes(hashlib.sha256(f"{serial}|{source_digest}".encode("ascii")).digest()[:8], "big")
    random = np.random.default_rng(rng_seed)
    for row in range(rows):
        for column in range(columns):
            y0 = row * BLOCK_SIZE
            x0 = column * BLOCK_SIZE
            dct = cv2.dct(luminance[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE])
            bit = bits[block_index(row, column, columns, len(bits))]
            dct[3, 2] = force_qim(dct[3, 2], bit)
            dct[2, 3] = force_qim(dct[2, 3], bit, 1.12)
            dct[6, 5] += random.choice(np.array([-1.0, 1.0], dtype=np.float32)) * 1.2
            dct[5, 6] += random.choice(np.array([-1.0, 1.0], dtype=np.float32)) * 1.2
            dct[7, 6] = force_mesh_qim(dct[7, 6], mesh[row * columns + column])
            luminance[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE] = cv2.idct(dct)
    ycrcb[:, :, 0] = np.clip(luminance, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), mesh_root


def decode_hidden_seal(image, serial, source_digest):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[:, :, 0].astype(np.float32)
    rows = luminance.shape[0] // BLOCK_SIZE
    columns = luminance.shape[1] // BLOCK_SIZE
    expected = payload_bits(serial, source_digest)
    votes = [[] for _ in range(len(expected))]
    for row in range(rows):
        for column in range(columns):
            y0 = row * BLOCK_SIZE
            x0 = column * BLOCK_SIZE
            dct = cv2.dct(luminance[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE])
            index = block_index(row, column, columns, len(expected))
            votes[index].append(int(round(abs(float(dct[3, 2])) / QIM_STEP)) % 2)
    recovered = np.array([int(np.mean(group) >= 0.5) if group else 0 for group in votes], dtype=np.uint8)
    confidence = float(np.mean(recovered == expected))
    return confidence, confidence >= 0.82


def decode_geometric_mesh(image, serial, source_digest):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[:, :, 0].astype(np.float32)
    rows = luminance.shape[0] // BLOCK_SIZE
    columns = luminance.shape[1] // BLOCK_SIZE
    expected, root = geometric_mesh(serial, source_digest, rows, columns)
    recovered = np.zeros(rows * columns, dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            y0 = row * BLOCK_SIZE
            x0 = column * BLOCK_SIZE
            dct = cv2.dct(luminance[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE])
            recovered[row * columns + column] = int(round(abs(float(dct[7, 6])) / MESH_QIM_STEP)) % 2
    confidence = float(np.mean(recovered == expected))
    return confidence, confidence >= 0.78, root


def encode_png(image, manifest):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("MEWS8", WATERMARK_VERSION)
    png_info.add_text("MEWS8-Manifest", base64.urlsafe_b64encode(canonical_json(manifest)).decode("ascii"))
    png_info.add_text("MEWS8-Signature", sign_manifest(manifest))
    output = BytesIO()
    Image.fromarray(rgb).save(output, format="PNG", pnginfo=png_info, compress_level=6)
    output.seek(0)
    return output


def read_manifest(data):
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG":
            raise ProcessingError("verification accepts PNG files only")
        encoded = image.text.get("MEWS8-Manifest")
        signature = image.text.get("MEWS8-Signature")
    if not encoded or not signature:
        raise ProcessingError("MEWS 8 provenance metadata is missing")
    try:
        manifest = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessingError("MEWS 8 provenance metadata is malformed") from error
    return manifest, signature


def seal(image_upload, watermark_upload):
    image = to_bgr(decode_upload(image_upload, "image"))
    watermark = decode_upload(watermark_upload, "watermark")
    source_digest = sha256(image.tobytes())
    serial = secrets.token_hex(8)
    branded = overlay_visible_mark(image, watermark)
    sealed, mesh_root = embed_hidden_seal(branded, serial, source_digest)
    manifest = {
        "version": WATERMARK_VERSION,
        "serial": serial,
        "sourceSha256": source_digest,
        "pixelSha256": sha256(sealed.tobytes()),
        "issuedAt": datetime.now(timezone.utc).isoformat(),
        "verificationNonceBits": format(int(serial, 16), "064b"),
        "redundancy": "distributed-8x8-qim",
        "geometricMesh": "cross-block-high-frequency",
        "meshRoot": mesh_root,
    }
    return encode_png(sealed, manifest), manifest


@app.post("/process")
def process():
    try:
        output, manifest = seal(request.files.get("image"), request.files.get("watermark"))
        response = send_file(output, mimetype="image/png", as_attachment=True, download_name=f"mews8-{manifest['serial']}.png")
        response.headers["X-MEWS8-Serial"] = manifest["serial"]
        response.headers["X-MEWS8-Version"] = WATERMARK_VERSION
        return response
    except ProcessingError as error:
        return jsonify(error=str(error)), 500
    except Exception:
        app.logger.exception("MEWS 8 processing failed")
        return jsonify(error="MEWS 8 could not process this image"), 500


@app.post("/verify")
def verify():
    try:
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            raise ProcessingError("missing image file")
        data = upload.read()
        manifest, provided_signature = read_manifest(data)
        expected_signature = sign_manifest(manifest)
        image = to_bgr(cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED))
        metadata_valid = hmac.compare_digest(provided_signature, expected_signature)
        pixels_valid = hmac.compare_digest(manifest.get("pixelSha256", ""), sha256(image.tobytes()))
        hidden_confidence, hidden_valid = decode_hidden_seal(image, manifest["serial"], manifest["sourceSha256"])
        mesh_confidence, mesh_valid, mesh_root = decode_geometric_mesh(image, manifest["serial"], manifest["sourceSha256"])
        mesh_root_valid = hmac.compare_digest(manifest.get("meshRoot", ""), mesh_root)
        return jsonify({
            "valid": metadata_valid and pixels_valid and hidden_valid and mesh_valid and mesh_root_valid,
            "metadataValid": metadata_valid,
            "pixelsValid": pixels_valid,
            "hiddenSealValid": hidden_valid,
            "hiddenSealConfidence": round(hidden_confidence, 4),
            "geometricMeshValid": mesh_valid and mesh_root_valid,
            "geometricMeshConfidence": round(mesh_confidence, 4),
            "serial": manifest.get("serial"),
            "version": manifest.get("version"),
        })
    except ProcessingError as error:
        return jsonify(error=str(error)), 500
    except Exception:
        app.logger.exception("MEWS 8 verification failed")
        return jsonify(error="MEWS 8 could not verify this image"), 500


@app.get("/health")
def health():
    return jsonify(service="MEWS 8", status="ok", version=WATERMARK_VERSION)


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return jsonify(error="upload exceeds the configured size limit"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
