# 🚗 ParkVision – Smart Parking Occupancy Detection System

## 1. Project Overview

ParkVision is an AI-based smart parking occupancy detection system that uses computer vision to automatically identify parking spaces and determine whether they are occupied or vacant.

The system uses a YOLOv8 model trained on the PKLot parking dataset and provides the detected results through a Flask-based web dashboard.

---

## 2. Problem Statement

Traditional parking systems often require manual monitoring or dedicated hardware to determine parking availability.

ParkVision aims to automatically analyze parking images and provide:

- Total detected parking spaces
- Occupied parking spaces
- Vacant parking spaces
- Occupancy percentage
- Visual identification of occupied and vacant spaces

---

## 3. Objectives

- Detect parking spaces automatically from an image.
- Classify each detected space as occupied or vacant.
- Calculate the overall parking occupancy rate.
- Display results through a web dashboard.
- Allow users to upload different parking images.
- Generate visual detection results automatically.

---

## 4. Proposed Solution

ParkVision uses a YOLOv8s model trained for PKLot parking-space detection.

The uploaded parking image is processed by the AI model. The model detects parking spaces and classifies them into two categories:

- **Occupied**
- **Vacant**

The detected results are then counted and displayed on the Flask dashboard.

Unlike the initial prototype, the current system does not depend on manually fixed parking-slot coordinates. The parking spaces are detected dynamically from the input image.

---

## 5. System Workflow

1. User uploads a parking image.
2. Flask receives the image.
3. The image is stored as the current input image.
4. YOLOv8s processes the image.
5. Parking spaces are detected automatically.
6. Each detected space is classified as occupied or vacant.
7. Occupied and vacant spaces are counted.
8. Occupancy percentage is calculated.
9. Bounding boxes are drawn on the image.
10. Results are stored in JSON format.
11. The processed image and statistics are displayed on the dashboard.

---

## 6. AI/ML Approach

### Model

YOLOv8s trained on the PKLot dataset is used for parking-space detection.

### Classes

- Class 0 → Vacant
- Class 1 → Occupied

### Confidence Threshold

The current detection confidence threshold is:

`0.40`

### IoU / NMS Threshold

The current IoU threshold is:

`0.40`

This helps reduce overlapping duplicate detections.

---

## 7. Occupancy Calculation

The occupancy rate is calculated as:

Occupancy Rate = (Occupied Spaces / Total Detected Spaces) × 100

For example, if 58 out of 110 detected spaces are occupied:

Occupancy Rate = (58 / 110) × 100 = 52.73%

---

## 8. Dashboard

The Flask dashboard displays:

- Total Parking Slots
- Occupied Slots
- Vacant Slots
- Occupancy Rate
- Occupancy Progress Bar
- AI Detection Result
- Occupied/Vacant visual indicators

The dashboard also supports uploading a new parking image and automatically running the detection process.

---

## 9. Technology Stack

### Programming Language
- Python

### AI/ML
- YOLOv8
- PKLot dataset
- Ultralytics

### Computer Vision
- OpenCV

### Web Development
- Flask
- HTML
- CSS
- JavaScript

### Data Storage
- JSON
- CSV

### Model Source
- Hugging Face

---

## 10. Project Structure

```text
ParkVision/
│
├── dataset/
│   └── images/
│       ├── parking.jpg
│       ├── parking2.jpg
│       ├── parking3.jpg
│       ├── parking4.jpg
│       ├── parking5.jpg
│       └── pklot_original.jpg
│
├── phase2_slots/
│   ├── occupancy.py
│   ├── dashboard.py
│   ├── test_system.py
│   ├── occupancy_data.json
│   ├── occupancy_result.jpg
│   ├── test_report.csv
│   │
│   ├── static/
│   │   └── occupancy_result.jpg
│   │
│   ├── templates/
│   │   └── dashboard.html
│   │
│   └── uploads/
│       └── parking.jpg
│
└── README.md
11. Testing and Evaluation

The system was tested using 6 image files.

One image (pklot_original.jpg) is a duplicate of parking.jpg, so there are effectively 5 unique test images.

Test Results
Image	Total Slots	Occupied	Vacant	Occupancy
parking.jpg	110	58	52	52.73%
parking2.jpg	142	20	122	14.08%
parking3.jpg	96	94	2	97.92%
parking4.jpg	116	114	2	98.28%
parking5.jpg	78	10	68	12.82%
pklot_original.jpg	110	58	52	52.73%

Average detection confidence across the tested images was approximately in the range of 0.696–0.813.

12. Important Evaluation Note

The above occupancy percentages represent the model's detected parking-space results.

They should not be treated as model accuracy, because ground-truth labels for these test images were not available.

Therefore, the current testing evaluates the system's detection output and consistency rather than calculating precision, recall, F1-score, or true classification accuracy.

13. Advantages
Automatic parking-space detection
No manually fixed slot coordinates
Supports different parking images
AI-based occupied/vacant classification
Web-based dashboard
Visual detection output
Easy image upload
Automatic occupancy calculation
14. Limitations
The model is trained on PKLot-style parking images.
Detection performance may decrease on images with significantly different camera angles or environments.
The current prototype processes images rather than continuous live video.
Ground-truth annotated data is required for proper accuracy evaluation.
15. Future Scope
Real-time CCTV/video processing
Automatic parking-space availability prediction
Vehicle tracking
Number-plate recognition
Parking navigation
Cloud deployment
Mobile application
Historical occupancy analytics
Database integration
Advanced model evaluation using precision, recall and F1-score
16. Current Project Status

The current prototype successfully includes:

Dynamic parking-space detection
Occupied/vacant classification
Occupancy calculation
Image upload
Flask dashboard
Detection visualization
JSON result storage
Automated system testing
CSV test report generation
17. How to Run

Activate the virtual environment:

.\venv\Scripts\Activate.ps1

Run the dashboard:

python phase2_slots\dashboard.py

Open:

http://127.0.0.1:5050/

Upload a parking image and click:

🔍 Upload & Analyze

18. Conclusion

ParkVision demonstrates an AI-based approach for automatically detecting parking-space occupancy using YOLOv8 and computer vision. The system dynamically detects parking spaces, classifies their occupancy status, calculates occupancy statistics, and presents the results through an interactive Flask dashboard.

The current prototype establishes the core functionality and provides a foundation for future real-time smart parking applications.