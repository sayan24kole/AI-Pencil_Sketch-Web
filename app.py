import base64
import io
import os
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)

# Max upload size: 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def safe_resize(image, max_dimension=1280):
    """
    Downscales large photos to a maximum dimension to ensure
    lightning-fast processing and zero RAM spikes on Render Free Tier.
    """
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def color_dodge(front, back):
    """
    Mathematical color dodge blending:
    Result = (Front * 256) / (255 - Back + epsilon)
    """
    result = cv2.divide(front, 255 - back, scale=256)
    return result


def synthesize_pencil_sketch(img_bgr, lead_grade=3, contrast=1.2, paper_texture=True):
    """
    Multi-stage Computational Sketch Engine:
    1. Edge-preserving smoothing (bilateral filter)
    2. Inversion & adaptive Gaussian blur kernel scaling
    3. Color-dodge contour isolation
    4. Lead-hardness tone mapping (2H -> 6B via dynamic gamma & gain)
    5. Graphite surface micro-texture overlay
    """
    # Step 0: Downscale safely if needed
    img_bgr = safe_resize(img_bgr, max_dimension=1280)

    # Step 1: Grayscale conversion
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: Edge-preserving filter (removes high-frequency skin noise while keeping facial contours)
    smooth_gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Step 3: Invert grayscale
    inverted_gray = 255 - smooth_gray

    # Step 4: Dynamically scale blur kernel with lead hardness
    # lead_grade ranges from 1 (2H Hard Lead) to 5 (6B Soft Dark Lead)
    kernel_sizes = {1: 15, 2: 21, 3: 31, 4: 45, 5: 61}
    k_size = kernel_sizes.get(int(lead_grade), 31)
    if k_size % 2 == 0:
        k_size += 1

    blurred_invert = cv2.GaussianBlur(inverted_gray, (k_size, k_size), sigmaX=0, sigmaY=0)

    # Step 5: Color dodge to extract pencil lines
    sketch = color_dodge(smooth_gray, blurred_invert)

    # Step 6: Lead Grade Tone Curve Modulation (Gamma Correction)
    # 2H (Hard): High gamma (faint, thin, technical lines)
    # 6B (Soft): Low gamma + gain (deep, rich graphite deposition)
    gamma_map = {1: 1.45, 2: 1.25, 3: 1.0, 4: 0.82, 5: 0.68}
    gamma = gamma_map.get(int(lead_grade), 1.0)

    # Non-linear gamma transformation
    inv_gamma = 1.0 / (gamma * float(contrast))
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    sketch = cv2.LUT(sketch, table)

    # Step 7: Dual-layer tonal shading (blends continuous tone back for realistic shading)
    # Mimics the tone decomposition method (Lu et al., 2012)
    tone_weight_map = {1: 0.05, 2: 0.12, 3: 0.22, 4: 0.32, 5: 0.42}
    tone_weight = tone_weight_map.get(int(lead_grade), 0.22)
    sketch = cv2.addWeighted(sketch, 1.0 - tone_weight, gray, tone_weight, 0)

    # Step 8: Optional graphite paper texture (adds subtle paper grain)
    if paper_texture:
        noise = np.random.normal(0, 3, sketch.shape).astype("float32")
        sketch_float = sketch.astype("float32") + noise
        sketch = np.clip(sketch_float, 0, 255).astype("uint8")

    return sketch


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/convert", methods=["POST"])
def convert_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Read parameters from request
        lead_grade = int(request.form.get("lead_grade", 3))  # 1=2H, 2=H, 3=HB, 4=2B, 5=6B
        contrast = float(request.form.get("contrast", 1.0))
        paper_texture = request.form.get("paper_texture", "true").lower() == "true"

        # Decode image in memory via NumPy (no disk I/O)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"error": "Could not decode uploaded image format"}), 400

        # Execute conversion
        sketch = synthesize_pencil_sketch(
            img_bgr,
            lead_grade=lead_grade,
            contrast=contrast,
            paper_texture=paper_texture,
        )

        # Encode resulting sketch to JPEG in-memory
        success, buffer = cv2.imencode(".jpg", sketch, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            return jsonify({"error": "Image encoding failed"}), 500

        sketch_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "status": "success",
            "sketch": f"data:image/jpeg;base64,{sketch_base64}",
            "dimensions": {"width": sketch.shape[1], "height": sketch.shape[0]},
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/healthz")
def health_check():
    """Healthcheck endpoint for Render zero-downtime health monitors"""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)