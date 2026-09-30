from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)


DATA_FILE = "phase2_slots/occupancy_data.json"


@app.route("/")
def home():

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    return render_template(
        "dashboard.html",
        data=data
    )


@app.route("/data")
def get_data():

    if not os.path.exists(DATA_FILE):
        return jsonify({
            "error": "Occupancy data not found"
        })

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    return jsonify(data)


if __name__ == "__main__":

    print("--------------------------------")
    print("ParkVision Dashboard")
    print("--------------------------------")

    print("Starting server...")
    print("Open: http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5050,
        debug=False,
        use_reloader=False
    )