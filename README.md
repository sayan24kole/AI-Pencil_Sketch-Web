
# SketchAI Studio ✏️

A lightweight web app that translates portraits into realistic pencil sketches in real time. Built with Flask, OpenCV, and Tailwind CSS, and optimized to run with a minimal memory (< 50MB RAM).

🔗 **Live Demo:** https://ai-pencil-sketch-web.onrender.com/

---

## Features

- **Pencil Lead Hardness Control ($2H \to 6B$):** Uses non-linear gamma curves to shift between faint technical drafting lines and deep, soft charcoal shading.
- **Interactive Split Comparison:** Slide smoothly between the original photo and the generated sketch.
- **Edge-Preserving Bilateral Smoothing:** Keeps critical facial contours sharp while removing sensor noise and skin blemishes.
- **Graphite & Paper Grain Simulation:** Overlays subtle high-frequency paper tooth texture.
- **Stateless & In-Memory:** Images are processed in-memory via NumPy buffers with zero disk writes, preventing server storage leaks.
- **One-Click HD Download:** Direct client-side JPEG export.

---

## Tech Stack

- **Backend:** Python, Flask, Gunicorn
- **Computer Vision:** OpenCV (`opencv-python-headless`), NumPy, Pillow
- **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript
- **Hosting:** Render (Free Web Service)

---

## Local Setup

If you want to run this project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<YOUR-USERNAME>/ai-pencil-sketch.git
   cd ai-pencil-sketch

2.  Create a virtual environment (optional but recommended):

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3.  Install dependencies:

    pip install -r requirements.txt

4.  Run the app:

    python app.py

    Open your browser and navigate to http://localhost:5000.

Project Structure

ai-pencil-sketch/
├── app.py              # Flask server & sketch rendering logic
├── Procfile            # Deployment command for Render
├── requirements.txt    # Minimal dependencies
├── templates/
│   └── index.html      # Frontend UI with comparison slider
└── README.md

References & Background

Inspired by recent work in conditional image-to-sketch translation (Pix2Pix++ /
U-Net++ architectures) and procedural tone decomposition methods (Lu et
al., 2012).

