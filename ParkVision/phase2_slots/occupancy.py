import cv2
import json
import os

from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# ==========================================
# FILES
# ==========================================

IMAGE_PATH = "dataset/images/parking.jpg"
OUTPUT_PATH = "phase2_slots/occupancy_result.jpg"
DATA_PATH = "phase2_slots/occupancy_data.json"

# Dashboard image
STATIC_OUTPUT_PATH = "phase2_slots/static/occupancy_result.jpg"


# ==========================================
# SETTINGS
# ==========================================

CONFIDENCE = 0.25


# ==========================================
# START
# ==========================================

print("--------------------------------")
print("ParkVision Parking Occupancy")
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

print(
    "Image size:",
    width,
    "x",
    height
)


# ==========================================
# LOAD PKLOT YOLO MODEL
# ==========================================

print("--------------------------------")
print("Loading PKLot YOLOv8s model...")
print("--------------------------------")


model_path = hf_hub_download(
    repo_id="dronefreak/pklot-yolov8s",
    filename="best.pt"
)

model = YOLO(model_path)


# ==========================================
# RUN MODEL
# ==========================================

print("--------------------------------")
print("Detecting parking spaces...")
print("--------------------------------")


results = model.predict(
    source=image,
    imgsz=960,
    conf=CONFIDENCE,
    verbose=False
)

result = results[0]


# ==========================================
# AUTOMATIC PARKING SLOT DETECTION
# ==========================================

slots = []


if result.boxes is not None:

    for box in result.boxes:

        coords = box.xyxy[0].tolist()

        x1, y1, x2, y2 = map(
            int,
            coords
        )

        confidence = float(
            box.conf[0]
        )

        class_id = int(
            box.cls[0]
        )


        # ==================================
        # PKLot classes
        # ==================================

        if class_id == 0:

            status = "VACANT"

        elif class_id == 1:

            status = "OCCUPIED"

        else:

            continue


        # ==================================
        # CREATE AUTOMATIC SLOT
        # ==================================

        slots.append({

            "box": [
                x1,
                y1,
                x2,
                y2
            ],

            "confidence": confidence,

            "status": status

        })


print(
    "Parking spaces detected:",
    len(slots)
)


# ==========================================
# CREATE OUTPUT IMAGE
# ==========================================

output = image.copy()


occupied_count = 0
vacant_count = 0


# ==========================================
# DRAW AUTOMATIC SLOTS
# ==========================================

for number, slot in enumerate(
    slots,
    start=1
):

    x1, y1, x2, y2 = slot["box"]

    status = slot["status"]

    confidence = slot["confidence"]


    # ======================================
    # COUNT
    # ======================================

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


    # ======================================
    # DRAW SLOT
    # ======================================

    cv2.rectangle(

        output,

        (x1, y1),

        (x2, y2),

        color,

        2

    )


    # ======================================
    # SLOT LABEL
    # ======================================

    label = (
        f"Slot {number}: "
        f"{status}"
    )


    cv2.putText(

        output,

        label,

        (
            x1,
            max(y1 - 5, 15)
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.35,

        color,

        1

    )


# ==========================================
# FINAL COUNTS
# ==========================================

total = len(slots)


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

print("================================")


# ==========================================
# SUMMARY ON IMAGE
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


# ==========================================
# CREATE OUTPUT DIRECTORIES
# ==========================================

os.makedirs(
    "phase2_slots",
    exist_ok=True
)

os.makedirs(
    "phase2_slots/static",
    exist_ok=True
)


# ==========================================
# SAVE RESULT IMAGE
# ==========================================

cv2.imwrite(

    OUTPUT_PATH,

    output

)


# Copy result for dashboard

cv2.imwrite(

    STATIC_OUTPUT_PATH,

    output

)


# ==========================================
# CREATE OCCUPANCY DATA
# ==========================================

occupancy_data = {

    "total_slots": total,

    "occupied": occupied_count,

    "vacant": vacant_count,

    "slot_status": [

        slot["status"]

        for slot in slots

    ],

    "slots": slots

}


# ==========================================
# SAVE JSON
# ==========================================

with open(

    DATA_PATH,

    "w"

) as f:

    json.dump(

        occupancy_data,

        f,

        indent=4

    )


print("Occupancy data saved.")

print(
    "Output:",
    OUTPUT_PATH
)

print("================================")
print("Detection completed.")
print("================================")