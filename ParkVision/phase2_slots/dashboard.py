from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
import shutil
import subprocess
import sys

app = Flask(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ==========================================
# FILE PATHS
# ==========================================

DATA_FILE = "phase2_slots/occupancy_data.json"

UPLOAD_FOLDER = "phase2_slots/uploads"

DATASET_IMAGE = "dataset/images/parking.jpg"

RESULT_IMAGE = "phase2_slots/occupancy_result.jpg"

STATIC_RESULT_IMAGE = "phase2_slots/static/occupancy_result.jpg"


# ==========================================
# CREATE REQUIRED FOLDERS
# ==========================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("phase2_slots/static", exist_ok=True)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    data = {
        "total_slots": 0,
        "occupied": 0,
        "vacant": 0,
        "occupancy_rate": 0,
        "slot_status": [],
        "slots": []
    }

    if os.path.exists(DATA_FILE):

        try:

            with open(DATA_FILE, "r") as f:
                data = json.load(f)

        except Exception as e:

            print("Error reading occupancy data:", e)

    return render_template(
        "dashboard.html",
        data=data
    )


# ==========================================
# API
# ==========================================

@app.route("/data")
def get_data():

    if not os.path.exists(DATA_FILE):

        return jsonify({
            "total_slots": 0,
            "occupied": 0,
            "vacant": 0,
            "occupancy_rate": 0
        })

    try:

        with open(DATA_FILE, "r") as f:
            data = json.load(f)

        return jsonify(data)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# IMAGE UPLOAD
# ==========================================

@app.route("/upload", methods=["POST"])
def upload_image():

    print()
    print("================================")
    print("NEW IMAGE UPLOAD")
    print("================================")

    if "image" not in request.files:
        return "No image selected", 400

    file = request.files["image"]

    if file.filename == "":
        return "No image selected", 400

    # Check file extension
    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:

        return (
            "Invalid file type. Please upload JPG, JPEG, or PNG image.",
            400
        )

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        "parking.jpg"
    )

    try:

        file.save(upload_path)

        print("Uploaded image:", upload_path)

        # Copy uploaded image to dataset folder
        shutil.copy2(
            upload_path,
            DATASET_IMAGE
        )

        print(
            "Copied image to:",
            DATASET_IMAGE
        )

        print()
        print("--------------------------------")
        print("Running AI parking detection...")
        print("--------------------------------")

        result = subprocess.run(
            [
                sys.executable,
                "phase2_slots/occupancy.py"
            ],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        print(result.stdout)

        if result.stderr:

            print("AI ERROR:")
            print(result.stderr)

        if result.returncode != 0:

            print("Detection failed.")

            return (
                "Image uploaded, but AI detection failed. "
                "Please upload a valid parking image.",
                500
            )

        if os.path.exists(RESULT_IMAGE):

            shutil.copy2(
                RESULT_IMAGE,
                STATIC_RESULT_IMAGE
            )

            print(
                "Updated dashboard result image."
            )

        else:

            print(
                "WARNING: Result image not found."
            )

        print("--------------------------------")
        print("AI detection completed")
        print("--------------------------------")

        return redirect(
            url_for("home")
        )

    except Exception as e:

        print(
            "Upload/detection error:",
            e
        )

        return (
            "Something went wrong while processing the image.",
            500
        )


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    print("--------------------------------")
    print("ParkVision Dashboard")
    print("--------------------------------")

    print(
        "Open: http://127.0.0.1:5050"
    )

    print("--------------------------------")

    app.run(
        host="127.0.0.1",
        port=5050,
        debug=False,
        use_reloader=False
    )