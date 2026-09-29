import cv2
import json
import os
import numpy as np
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "images",
    "parking.jpg"
)

SLOTS_PATH = os.path.join(
    BASE_DIR,
    "phase2_slots",
    "slots.json"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "phase2_slots",
    "occupancy_result.jpg"
)


# ============================================================
# SETTINGS
# ============================================================

# YOLO confidence
YOLO_CONFIDENCE = 0.05

# Enlarge every parking-slot crop
SCALE = 4

# Vehicle classes
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# If YOLO finds a vehicle with this confidence,
# consider the slot occupied.
YOLO_SLOT_CONFIDENCE = 0.05

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(IMAGE_PATH):
    print("ERROR: Image not found!")
    print("Expected:", IMAGE_PATH)
    exit()

if not os.path.exists(SLOTS_PATH):
    print("ERROR: slots.json not found!")
    print("Expected:", SLOTS_PATH)
    exit()


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not read parking image!")
    exit()


height, width = image.shape[:2]

print("--------------------------------")
print("Parking Occupancy Detection")
print("--------------------------------")
print("Image:", IMAGE_PATH)
print(
    "Image size:",
    width,
    "x",
    height
)


# ============================================================
# LOAD SLOTS
# ============================================================

with open(SLOTS_PATH, "r") as f:
    slots = json.load(f)

print("Total slots:", len(slots))


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLOv8s...")

model = YOLO("yolov8s.pt")


# ============================================================
# OUTPUT IMAGE
# ============================================================

output = image.copy()


# ============================================================
# PROCESS EACH PARKING SLOT
# ============================================================

occupied_count = 0


for slot_number, slot in enumerate(slots, start=1):

    print("--------------------------------")
    print("Checking Slot", slot_number)
    print("--------------------------------")


    # --------------------------------------------------------
    # Convert slot coordinates
    # --------------------------------------------------------

    polygon = np.array(
        slot,
        dtype=np.int32
    )


    # --------------------------------------------------------
    # Find bounding rectangle of slot
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        polygon
    )


    # Keep coordinates inside image
    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(
        width,
        x + w
    )

    y2 = min(
        height,
        y + h
    )


    # --------------------------------------------------------
    # Crop slot
    # --------------------------------------------------------

    crop = image[
        y1:y2,
        x1:x2
    ]


    if crop.size == 0:

        print("Invalid slot crop")
        continue


    # --------------------------------------------------------
    # Create mask for polygon
    # --------------------------------------------------------

    local_polygon = polygon.copy()

    local_polygon[:, 0] -= x1
    local_polygon[:, 1] -= y1


    mask = np.zeros(
        crop.shape[:2],
        dtype=np.uint8
    )


    cv2.fillPoly(
        mask,
        [local_polygon],
        255
    )


    # --------------------------------------------------------
    # Keep only slot region
    # --------------------------------------------------------

    slot_crop = cv2.bitwise_and(
        crop,
        crop,
        mask=mask
    )


    # --------------------------------------------------------
    # Enlarge slot
    # --------------------------------------------------------

    enlarged = cv2.resize(
        slot_crop,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_CUBIC
    )


    # ========================================================
    # YOLO DETECTION ON INDIVIDUAL SLOT
    # ========================================================

    results = model.predict(

        source=enlarged,

        imgsz=640,

        conf=YOLO_CONFIDENCE,

        classes=VEHICLE_CLASSES,

        verbose=False

    )


    result = results[0]


    best_confidence = 0

    detected_vehicle = False


    if result.boxes is not None:

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )


            if confidence > best_confidence:

                best_confidence = confidence


            if confidence >= YOLO_SLOT_CONFIDENCE:

                detected_vehicle = True


    # ========================================================
    # OCCUPANCY DECISION
    # ========================================================

    if detected_vehicle:

        occupied = True

    else:

        occupied = False


    # ========================================================
    # RESULT
    # ========================================================

    if occupied:

        occupied_count += 1

        color = (
            0,
            0,
            255
        )

        status = "OCCUPIED"

        print(
            "Vehicle detected!"
        )

        print(
            "Confidence:",
            round(
                best_confidence,
                3
            )
        )

    else:

        color = (
            0,
            255,
            0
        )

        status = "VACANT"

        print(
            "No vehicle detected."
        )

        print(
            "Best confidence:",
            round(
                best_confidence,
                3
            )
        )


    # ========================================================
    # DRAW SLOT
    # ========================================================

    cv2.polylines(

        output,

        [polygon],

        True,

        color,

        4

    )


    # --------------------------------------------------------
    # Slot label position
    # --------------------------------------------------------

    label_x = x

    label_y = max(
        25,
        y - 8
    )


    cv2.putText(

        output,

        f"Slot {slot_number}: {status}",

        (
            label_x,
            label_y
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        color,

        2,

        cv2.LINE_AA

    )


# ============================================================
# FINAL COUNTS
# ============================================================

total_slots = len(slots)

vacant_count = (
    total_slots -
    occupied_count
)


# ============================================================
# SUMMARY ON IMAGE
# ============================================================

cv2.rectangle(

    output,

    (10, 10),

    (300, 140),

    (0, 0, 0),

    -1

)


cv2.putText(

    output,

    f"Total Slots: {total_slots}",

    (25, 45),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.75,

    (255, 255, 255),

    2

)


cv2.putText(

    output,

    f"Occupied: {occupied_count}",

    (25, 80),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.75,

    (0, 0, 255),

    2

)


cv2.putText(

    output,

    f"Vacant: {vacant_count}",

    (25, 115),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.75,

    (0, 255, 0),

    2

)


# ============================================================
# SAVE OUTPUT
# ============================================================

cv2.imwrite(

    OUTPUT_PATH,

    output

)


# ============================================================
# FINAL TERMINAL OUTPUT
# ============================================================

print()
print("================================")
print("FINAL RESULT")
print("================================")

print(
    "Total slots :",
    total_slots
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
    "Output      :",
    OUTPUT_PATH
)

print("================================")


# ============================================================
# SHOW OUTPUT
# ============================================================

cv2.imshow(

    "ParkVision - Parking Occupancy",

    output

)

print(
    "Press any key on the image window to close."
)

cv2.waitKey(0)

cv2.destroyAllWindows()