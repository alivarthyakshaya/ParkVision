import cv2
import os
import csv

from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# ==========================================
# SETTINGS
# ==========================================

IMAGE_FOLDER = "dataset/images"

CONFIDENCE = 0.40
NMS_IOU = 0.40

REPORT_FILE = "phase2_slots/test_report.csv"


# ==========================================
# LOAD MODEL
# ==========================================

print("--------------------------------")
print("ParkVision System Testing")
print("--------------------------------")

print("Loading PKLot YOLOv8s model...")

model_path = hf_hub_download(
    repo_id="dronefreak/pklot-yolov8s",
    filename="best.pt"
)

model = YOLO(model_path)

print("Model loaded.")
print()


# ==========================================
# FIND IMAGES
# ==========================================

images = []

for filename in os.listdir(IMAGE_FOLDER):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        images.append(filename)


print(
    "Images found:",
    len(images)
)

print()


# ==========================================
# REPORT DATA
# ==========================================

report = []


# ==========================================
# TEST EACH IMAGE
# ==========================================

for number, filename in enumerate(
    images,
    start=1
):

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:

        print(
            "Could not read:",
            filename
        )

        continue


    print("--------------------------------")
    print(
        f"Test {number}: {filename}"
    )
    print("--------------------------------")


    # ======================================
    # RUN YOLO
    # ======================================

    results = model.predict(

        source=image,

        imgsz=960,

        conf=CONFIDENCE,

        iou=NMS_IOU,

        verbose=False

    )

    result = results[0]


    # ======================================
    # COUNTS
    # ======================================

    total = 0
    occupied = 0
    vacant = 0

    confidence_values = []


    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            confidence_values.append(
                confidence
            )


            if class_id == 0:

                vacant += 1

            elif class_id == 1:

                occupied += 1

            else:

                continue


            total += 1


    # ======================================
    # OCCUPANCY RATE
    # ======================================

    if total > 0:

        occupancy_rate = (
            occupied / total
        ) * 100

    else:

        occupancy_rate = 0


    # ======================================
    # AVERAGE CONFIDENCE
    # ======================================

    if confidence_values:

        average_confidence = (
            sum(confidence_values)
            /
            len(confidence_values)
        )

    else:

        average_confidence = 0


    # ======================================
    # PRINT
    # ======================================

    print(
        "Total slots :",
        total
    )

    print(
        "Occupied    :",
        occupied
    )

    print(
        "Vacant      :",
        vacant
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
        "Avg confidence:",
        round(
            average_confidence,
            3
        )
    )


    # ======================================
    # SAVE REPORT
    # ======================================

    report.append({

        "image": filename,

        "total_slots": total,

        "occupied": occupied,

        "vacant": vacant,

        "occupancy_rate":
            round(
                occupancy_rate,
                2
            ),

        "average_confidence":
            round(
                average_confidence,
                3
            )

    })


print()
print("================================")
print("TESTING COMPLETED")
print("================================")


# ==========================================
# SAVE CSV REPORT
# ==========================================

with open(
    REPORT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=[
            "image",
            "total_slots",
            "occupied",
            "vacant",
            "occupancy_rate",
            "average_confidence"
        ]

    )

    writer.writeheader()

    writer.writerows(report)


print(
    "Report saved:",
    REPORT_FILE
)

print("================================")