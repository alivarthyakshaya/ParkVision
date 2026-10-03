from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
import shutil
import subprocess
import sys

app = Flask(__name__)

DATA_FILE = "phase2_slots/occupancy_data.json"
UPLOAD_FOLDER = "phase2_slots/uploads"
RESULT_IMAGE = "phase2_slots/occupancy_result.jpg"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():

    if not os.path.exists(DATA_FILE):
        data = {
            "total_slots": 0,
            "occupied": 0,
            "vacant": 0,
            "slots": []
        }
    else:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)

    return render_template("dashboard.html", data=data)


@app.route("/data")
def get_data():

    if not os.path.exists(DATA_FILE):
        return jsonify({
            "error": "Occupancy data not found"
        })

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    return jsonify(data)


@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return "No image selected", 400

    file = request.files["image"]

    if file.filename == "":
        return "No image selected", 400

    # Create upload folder
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Save uploaded image
    upload_path = os.path.join(
        UPLOAD_FOLDER,
        "parking.jpg"
    )

    file.save(upload_path)

    # Copy image to dataset location
    dataset_image = "dataset/images/parking.jpg"

    os.makedirs("dataset/images", exist_ok=True)

    shutil.copy(
        upload_path,
        dataset_image
    )

    print("--------------------------------")
    print("New parking image uploaded")
    print("Running occupancy detection...")
    print("--------------------------------")

    # Run occupancy detection script
    result = subprocess.run(
        [sys.executable, "phase2_slots/occupancy.py"],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print("ERROR:")
        print(result.stderr)

    print("--------------------------------")
    print("Detection completed")
    print("--------------------------------")

    return redirect(url_for("home"))


if __name__ == "__main__":

    print("--------------------------------")
    print("ParkVision Dashboard")
    print("--------------------------------")
    print("Starting server...")
    print("Open: http://127.0.0.1:5050")

    app.run(
        host="127.0.0.1",
        port=5050,
        debug=False,
        use_reloader=False
    )