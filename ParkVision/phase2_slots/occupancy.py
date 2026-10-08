import cv2
import json
import os
from datetime import datetime

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ==========================================
# FILE PATHS
# ==========================================

IMAGE_PATH = "dataset/images/parking.jpg"

OUTPUT_PATH = "phase2_slots/occupancy_result.jpg"

DATA_PATH = "phase2_slots/occupancy_data.json"

HISTORY_PATH = "phase2_slots/occupancy_history.json"

STATIC_OUTPUT_PATH = "phase2_slots/static/occupancy_result.jpg"


# ==========================================
# SETTINGS
# ==========================================

CONFIDENCE = 0.40

NMS_IOU = 0.40

LOW_CONFIDENCE_THRESHOLD = 0.55

MAX_HISTORY = 100


# ==========================================
# HEADER
# ==========================================

print("--------------------------------")
print("ParkVision Smart Parking System")
print("--------------------------------")

print("Image:", IMAGE_PATH)


# ==========================================
# LOAD IMAGE
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:

    print("ERROR: parking.jpg not found!")

    exit()


height, width = image.shape[:2]

print("Image size:", width, "x", height)


# ==========================================
# LOAD MODEL
# ==========================================

print("--------------------------------")
print("Loading PKLot YOLOv8s model...")
print("--------------------------------")

model_path = hf_hub_download(
    repo_id="dronefreak/pklot-yolov8s",
    filename="best.pt"
)

model = YOLO(model_path)
print("Model loaded.")
# ==========================================
# DETECTION
# ==========================================
print("--------------------------------")
print("Detecting parking spaces...")
print("--------------------------------")
results = model.predict(source=image,imgsz=960,conf=CONFIDENCE,iou=NMS_IOU,verbose=False)
result = results[0]
# ==========================================
# PROCESS DETECTIONS
# ==========================================
slots = []
if result.boxes is not None:
    for box in result.boxes:
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        # ----------------------------------
        # CLASSIFICATION
        # ----------------------------------
        if class_id == 0:
            status = "VACANT"
        elif class_id == 1:
            status = "OCCUPIED"
        else:
            continue
        # ----------------------------------
        # CONFIDENCE ANALYSIS
        # ----------------------------------
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            confidence_status = "LOW"
        else:
            confidence_status = "HIGH"
        slots.append({
            "box": [
                x1,
                y1,
                x2,
                y2
            ],

            "confidence": round(
                confidence,
                3
            ),

            "status": status,

            "confidence_status":
                confidence_status

        })


# ==========================================
# COUNTS
# ==========================================

print(
    "Parking spaces detected:",
    len(slots)
)


output = image.copy()


occupied_count = 0

vacant_count = 0

low_confidence_count = 0


# ==========================================
# DRAW DETECTIONS
# ==========================================

for number, slot in enumerate(
    slots,
    start=1
):

    x1, y1, x2, y2 = slot["box"]

    status = slot["status"]

    confidence = slot["confidence"]

    confidence_status = \
        slot["confidence_status"]


    # ----------------------------------
    # COUNT
    # ----------------------------------

    if status == "OCCUPIED":

        occupied_count += 1

        color = (
            0,
            0,
            255
        )

    else:

        vacant_count += 1

        color = (
            0,
            255,
            0
        )


    # ----------------------------------
    # LOW CONFIDENCE
    # ----------------------------------

    if confidence_status == "LOW":

        low_confidence_count += 1

        # Orange warning border

        border_color = (
            0,
            165,
            255
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            border_color,
            3
        )

    else:

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


    # ----------------------------------
    # SLOT NUMBER
    # ----------------------------------

    cv2.putText(
        output,
        str(number),
        (x1 + 2, y1 + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1
    )


    # ----------------------------------
    # CONFIDENCE
    # ----------------------------------

    confidence_text = (
        f"{confidence * 100:.0f}%"
    )

    cv2.putText(
        output,
        confidence_text,
        (x1, y2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1
    )


# ==========================================
# OCCUPANCY CALCULATION
# ==========================================

total = len(slots)


if total > 0:

    occupancy_rate = (
        occupied_count /
        total
    ) * 100

else:

    occupancy_rate = 0


# ==========================================
# SMART RECOMMENDATION
# ==========================================

vacant_slots = []

for number, slot in enumerate(
    slots,
    start=1
):

    if slot["status"] == "VACANT":

        vacant_slots.append({

            "slot": number,

            "confidence":
                slot["confidence"]

        })


# Highest-confidence vacant slots first

vacant_slots.sort(
    key=lambda x:
        x["confidence"],
    reverse=True
)


recommended_slots = [
    item["slot"]
    for item in vacant_slots[:5]
]


# ==========================================
# LOAD HISTORY
# ==========================================

history = []


if os.path.exists(HISTORY_PATH):

    try:

        with open(
            HISTORY_PATH,
            "r"
        ) as f:

            history = json.load(f)

    except Exception:

        history = []


# ==========================================
# ADD CURRENT RESULT TO HISTORY
# ==========================================

analysis_time = datetime.now().strftime(
    "%d-%m-%Y %I:%M:%S %p"
)


history.append({

    "analysis_time":
        analysis_time,

    "total_slots":
        total,

    "occupied":
        occupied_count,

    "vacant":
        vacant_count,

    "occupancy_rate":
        round(
            occupancy_rate,
            2
        )

})


# Keep latest 100 analyses

history = history[-MAX_HISTORY:]


# ==========================================
# SAVE HISTORY
# ==========================================

os.makedirs(
    "phase2_slots",
    exist_ok=True
)


with open(
    HISTORY_PATH,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ==========================================
# DEMAND PREDICTION
# ==========================================

predicted_occupancy = None

prediction_message = (
    "Not enough history for prediction."
)


if len(history) >= 3:

    recent_rates = [

        item["occupancy_rate"]

        for item in history

    ]


    # Simple trend calculation

    differences = []

    for i in range(
        1,
        len(recent_rates)
    ):

        differences.append(
            recent_rates[i]
            -
            recent_rates[i - 1]
        )


    average_change = (
        sum(differences)
        /
        len(differences)
    )


    predicted_occupancy = (
        occupancy_rate
        +
        average_change
    )


    # Keep prediction within 0-100

    predicted_occupancy = max(
        0,
        min(
            100,
            predicted_occupancy
        )
    )


    if predicted_occupancy > \
            occupancy_rate + 2:

        prediction_message = (
            "Parking demand is expected "
            "to increase."
        )

    elif predicted_occupancy < \
            occupancy_rate - 2:

        prediction_message = (
            "Parking demand is expected "
            "to decrease."
        )

    else:

        prediction_message = (
            "Parking demand is expected "
            "to remain stable."
        )


# ==========================================
# PARKING AVAILABILITY STATUS
# ==========================================

if occupancy_rate < 50:

    availability_status = \
        "PLENTY OF PARKING"

elif occupancy_rate <= 80:

    availability_status = \
        "LIMITED PARKING"

else:

    availability_status = \
        "NEARLY FULL"


# ==========================================
# PRINT FINAL RESULT
# ==========================================

print()

print("================================")
print("FINAL RESULT")
print("================================")

print(
    "Total slots :",
    total
)

print(
    "Occupied    :",
    occupied_count
)

print(
    "Vacant      :",
    vacant_count
)

print(
    "Occupancy   :",
    round(
        occupancy_rate,
        2
    ),
    "%"
)

print(
    "Low confidence detections:",
    low_confidence_count
)

print(
    "Recommended slots:",
    recommended_slots
)

if predicted_occupancy is not None:

    print(
        "Predicted occupancy:",
        round(
            predicted_occupancy,
            2
        ),
        "%"
    )

print(
    "Status:",
    availability_status
)

print("================================")


# ==========================================
# DRAW SUMMARY
# ==========================================

cv2.putText(
    output,
    f"Occupied: {occupied_count}",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2
)

cv2.putText(
    output,
    f"Vacant: {vacant_count}",
    (20, 65),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

cv2.putText(
    output,
    f"Total Slots: {total}",
    (20, 95),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.putText(
    output,
    f"Occupancy: {occupancy_rate:.1f}%",
    (20, 125),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.putText(
    output,
    f"Low Confidence: {low_confidence_count}",
    (20, 155),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 165, 255),
    2
)


# ==========================================
# SAVE OUTPUT IMAGE
# ==========================================

os.makedirs(
    "phase2_slots/static",
    exist_ok=True
)

cv2.imwrite(
    OUTPUT_PATH,
    output
)

cv2.imwrite(
    STATIC_OUTPUT_PATH,
    output
)


# ==========================================
# SAVE CURRENT DATA
# ==========================================

occupancy_data = {

    "analysis_time":
        analysis_time,

    "total_slots":
        total,

    "occupied":
        occupied_count,

    "vacant":
        vacant_count,

    "occupancy_rate":
        round(
            occupancy_rate,
            2
        ),

    "availability_status":
        availability_status,

    "low_confidence_count":
        low_confidence_count,

    "recommended_slots":
        recommended_slots,

    "predicted_occupancy":
        (
            round(
                predicted_occupancy,
                2
            )
            if predicted_occupancy is not None
            else None
        ),

    "prediction_message":
        prediction_message,

    "slot_status":
        [
            slot["status"]
            for slot in slots
        ],

    "slots":
        slots

}


with open(
    DATA_PATH,
    "w"
) as f:

    json.dump(
        occupancy_data,
        f,
        indent=4
    )


# ==========================================
# FINISH
# ==========================================

print()

print(
    "Occupancy data saved."
)

print(
    "History saved:",
    HISTORY_PATH
)

print(
    "Output:",
    OUTPUT_PATH
)

print("--------------------------------")
print("Detection completed.")
print("--------------------------------")