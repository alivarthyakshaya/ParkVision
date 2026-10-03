# 🚗 ParkVision – Smart Parking Occupancy Detection System

ParkVision is an AI-based smart parking system that automatically detects parking spaces and identifies whether each parking slot is **occupied or vacant** from a parking-lot image.

The system uses a **YOLOv8-based parking-space detection model**, parking-slot analysis using **Intersection over Union (IoU)**, and a **Flask web dashboard** to display the parking occupancy information.

---

## 📌 Project Overview

Finding an available parking space in crowded parking areas can be time-consuming and inconvenient.

ParkVision provides an automated solution by:

1. Taking a parking-lot image as input.
2. Detecting parking spaces using a YOLOv8-based model.
3. Comparing detected parking areas with predefined parking slots.
4. Calculating IoU to determine occupancy.
5. Classifying each slot as **OCCUPIED** or **VACANT**.
6. Saving the results in a JSON file.
7. Displaying the results through a web-based Flask dashboard.

---

## 🎯 Objectives

- Automatically detect parking spaces.
- Determine the occupancy status of individual parking slots.
- Reduce the need for manual parking-space monitoring.
- Provide a visual representation of parking occupancy.
- Display parking statistics through a web dashboard.
- Create a foundation for a real-time smart parking system.

---

## 🏗️ System Architecture

```text
                Parking Lot Image
                       │
                       ▼
             YOLOv8 Parking Model
                       │
                       ▼
             Parking Space Detection
                       │
                       ▼
             Predefined Parking Slots
                       │
                       ▼
                IoU Calculation
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
         OCCUPIED             VACANT
             │                   │
             └─────────┬─────────┘
                       ▼
              occupancy_data.json
                       │
                       ▼
                  Flask API
                    /data
                       │
                       ▼
              ParkVision Dashboard
                       │
                       ▼
              Occupancy Visualization
🧠 AI/ML Approach
YOLOv8

ParkVision uses a YOLOv8-based model trained/selected for parking-space detection.

The model identifies parking-space regions from the input image.

The detected regions are then analyzed against the predefined parking-slot coordinates.

📐 IoU-Based Occupancy Detection

Intersection over Union (IoU) is used to measure the overlap between a predefined parking slot and a detected parking region.

             Area of Intersection
IoU = ------------------------------------
             Area of Union

A suitable IoU value is used along with detection confidence to determine whether a parking slot is occupied.

The current system produces results such as:

Slot 1 : OCCUPIED
Slot 2 : VACANT
Slot 3 : OCCUPIED
...
🅿️ Parking Slot Detection

The current prototype uses predefined parking-slot coordinates.

Example:

{
    "slots": [
        [373, 369, 419, 430],
        [447, 377, 480, 429],
        [526, 267, 563, 331],
        [248, 367, 273, 428],
        [384, 275, 409, 322],
        [205, 276, 228, 317],
        [319, 275, 349, 329]
    ]
}

Each coordinate represents a parking-slot bounding region in the input image.

📊 Current Detection Result

For the current test image, the system detected:

Total Slots : 7
Occupied    : 6
Vacant      : 1

Therefore:

Occupancy Rate = 85.7%

The result is stored in:

phase2_slots/occupancy_data.json
🌐 Web Dashboard

ParkVision includes a Flask-based web dashboard.

The dashboard displays:

Total parking slots
Occupied slots
Vacant slots
Occupancy percentage
Parking detection image
Individual parking-slot status
Refresh functionality
Automatic data updates
🖥️ Dashboard Features
1. Total Slots

Displays the total number of monitored parking spaces.

2. Occupied Slots

Displays the number of currently occupied spaces.

3. Vacant Slots

Displays the number of available spaces.

4. Occupancy Rate

The dashboard calculates:

Occupancy Rate =
(Occupied Slots / Total Slots) × 100
5. Parking Detection Image

The dashboard displays the processed parking image with detected parking spaces.

6. Slot Status

Each parking slot is displayed individually as:

Slot 1 → OCCUPIED
Slot 2 → VACANT
Slot 3 → OCCUPIED
7. Automatic Dashboard Updates

The dashboard checks the Flask /data endpoint every 5 seconds and updates the displayed statistics.

🛠️ Technology Stack
Programming Language
Python
Machine Learning
YOLOv8
Computer Vision
Intersection over Union (IoU)
Backend
Flask
Frontend
HTML
CSS
JavaScript
Data Storage
JSON
Dataset
PKLot-based parking dataset
📂 Project Structure
ParkVision/
│
├── dataset/
│   └── images/
│       ├── parking.jpg
│       └── pklot_original.jpg
│
├── phase2_slots/
│   │
│   ├── occupancy.py
│   ├── occupancy_data.json
│   ├── occupancy_result.jpg
│   │
│   ├── static/
│   │   └── occupancy_result.jpg
│   │
│   └── templates/
│       └── dashboard.html
│
├── README.md
│
└── venv/
📄 Important Files
occupancy.py

Responsible for:

Loading the parking image.
Loading the parking detection model.
Detecting parking spaces.
Comparing detected regions with predefined slots.
Calculating IoU.
Determining occupancy.
Generating the output image.
Saving occupancy information.
occupancy_data.json

Stores the detected parking information.

Example:

{
    "total_slots": 7,
    "occupied": 6,
    "vacant": 1
}
dashboard.py

Flask backend responsible for:

Starting the web server.
Loading occupancy data.
Rendering the dashboard.
Providing the /data API endpoint.
dashboard.html

Frontend interface responsible for displaying:

Parking statistics
Occupancy percentage
Parking detection image
Individual slot statuses
Automatic updates
⚙️ Installation
1. Clone or open the project

Open the project directory:

C:\Users\Akshaya\Documents\GitHub\ParkVision\ParkVision
2. Create/activate the virtual environment

Activate the virtual environment:

venv\Scripts\activate

You should see:

(venv)

at the beginning of the PowerShell prompt.

3. Install required packages

Install Flask:

pip install flask

Install the required computer-vision/ML packages according to the model used by the project.

▶️ How to Run
Step 1 – Run Occupancy Detection

From the project root:

python phase2_slots\occupancy.py

The program processes the parking image and generates the occupancy information.

Step 2 – Start the Dashboard

Run:

python phase2_slots\dashboard.py

The Flask server starts at:

http://127.0.0.1:5050/

Open this address in a web browser.

🔄 Complete Workflow

The complete ParkVision workflow is:

1. Input Parking Image
          ↓
2. Load YOLOv8 Parking Model
          ↓
3. Detect Parking Spaces
          ↓
4. Load Predefined Parking Slots
          ↓
5. Calculate IoU
          ↓
6. Determine Occupancy
          ↓
7. Generate Occupancy Result Image
          ↓
8. Save occupancy_data.json
          ↓
9. Flask Loads JSON Data
          ↓
10. Dashboard Displays Results
          ↓
11. Dashboard Updates Data Automatically
📈 Example Output

For the current test image:

--------------------------------
Parking Occupancy Result
--------------------------------

Total slots : 7
Occupied    : 6
Vacant      : 1

Occupancy rate : 85.7%

The dashboard then displays the same information visually.

🔌 Flask API

ParkVision provides a data endpoint:

GET /data

When the dashboard requests:

http://127.0.0.1:5050/data

the Flask server returns the latest occupancy information in JSON format.

This allows the frontend to update the statistics without manually reloading the entire page.

📊 Dashboard Data Flow
occupancy_data.json
        │
        ▼
     Flask
        │
        ▼
     /data API
        │
        ▼
   JavaScript fetch()
        │
        ▼
 Dashboard Statistics

The dashboard currently checks the /data endpoint every 5 seconds.

🎯 Current Project Status
Feature	Status
PKLot-based image	✅ Completed
Parking-space detection	✅ Completed
Parking-slot coordinates	✅ Completed
IoU-based occupancy analysis	✅ Completed
Occupied/Vacant classification	✅ Completed
JSON occupancy data	✅ Completed
Flask backend	✅ Completed
Web dashboard	✅ Completed
Parking result image	✅ Completed
Automatic dashboard update	✅ Completed
Real-time camera input	🔄 Future work
Live parking-space reservation	🔄 Future work
Multi-camera support	🔄 Future work
🚀 Future Scope

The current prototype can be extended into a complete smart parking platform.

1. Real-Time Camera Integration

Connect CCTV/IP cameras or video streams to continuously monitor parking spaces.

2. Automatic Slot Detection

Instead of manually defining parking-slot coordinates, a future version can automatically identify parking spaces.

3. Live Occupancy Monitoring

The system can continuously update parking availability from live camera feeds.

4. Parking Availability Search

Users can be shown the currently available parking-slot numbers.

Example:

Available Slots:
Slot 2
Slot 5
Slot 7
5. Multiple Parking Areas

The system can monitor multiple parking lots or floors.

6. Database Integration

JSON storage can be replaced or extended with a database such as MySQL or MongoDB.

7. Mobile Application

A mobile application can provide parking availability information to users.

8. Historical Analytics

The system can store historical occupancy data and generate:

Daily occupancy
Peak parking hours
Average occupancy
Parking utilization trends
🔐 Limitations

The current prototype has some limitations:

The current demonstration uses a parking image rather than a live camera stream.
Parking-slot coordinates are currently predefined.
Detection performance depends on image quality, camera angle, lighting, and model performance.
The current implementation is a prototype and is not yet connected to a real parking-management system.
💡 Advantages
Automated parking-space monitoring
Reduces manual observation
Provides visual occupancy information
Easy-to-use web dashboard
Can be extended to real-time camera systems
Supports further AI-based parking analytics
🧪 Testing

The system can be tested using parking-lot images.

Testing involves:

Providing a parking image.
Running the occupancy detection program.
Checking detected parking spaces.
Checking occupied/vacant classifications.
Verifying occupancy_data.json.
Starting the Flask dashboard.
Comparing dashboard results with the generated detection image.
📝 Conclusion

ParkVision demonstrates an AI-based approach for automated parking occupancy detection.

By combining YOLOv8-based computer vision, IoU-based parking analysis, JSON data processing, and a Flask web dashboard, the system can identify occupied and vacant parking spaces and present the information in an easy-to-understand interface.

The current prototype provides the foundation for developing a more advanced real-time smart parking management system.