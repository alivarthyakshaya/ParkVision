import cv2
import json
import os
import numpy as np
from ultralytics import YOLO


# ============================================================
# PARKVISION - PARKING OCCUPANCY DETECTION
# ============================================================

# ---------- FILE PATHS ----------
IMAGE_PATH = "dataset/images/parking.jpg"
SLOTS_PATH = "phase2_slots/slots.json"
OUTPUT_PATH = "phase2_slots/occupancy_result.jpg"
MODEL_PATH = "yolov8n.pt"


# ---------- SETTINGS ----------
CONFIDENCE_THRESHOLD = 0.10

# Percentage of the parking-slot area that must overlap
# with a detected vehicle to consider the slot occupied.
OCCUPANCY_THRESHOLD = 0.10


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(IMAGE_PATH):
    print("ERROR: Image not found!")
    print("Expected:", os.path.abspath(IMAGE_PATH))
    exit()

if not os.path.exists(SLOTS_PATH):
    print("ERROR: slots.json not found!")
    print("Expected:", os.path.abspath(SLOTS_PATH))
    exit()


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not read image!")
    exit()


original = image.copy()

height, width = image.shape[:2]


# ============================================================
# LOAD PARKING SLOTS
# ============================================================

with open(SLOTS_PATH, "r") as file:
    slots = json.load(file)


print("--------------------------------")
print("Parking Occupancy Detection")
print("--------------------------------")
print("Total slots :", len(slots))


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# VEHICLE DETECTION
# ============================================================

results = model(
    image,
    conf=CONFIDENCE_THRESHOLD,
    imgsz=1280,
    verbose=False
)


# COCO vehicle classes
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

vehicle_classes = [2, 3, 5, 7]

vehicle_boxes = []


for result in results:

    if result.boxes is None:
        continue

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        vehicle_boxes.append(
            (x1, y1, x2, y2, class_id, confidence)
        )


print("Vehicles detected:", len(vehicle_boxes))


# ============================================================
# DRAW VEHICLE DETECTIONS
# ============================================================

for x1, y1, x2, y2, class_id, confidence in vehicle_boxes:

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    if class_id == 2:
        label = "Car"

    elif class_id == 3:
        label = "Motorcycle"

    elif class_id == 5:
        label = "Bus"

    elif class_id == 7:
        label = "Truck"

    else:
        label = "Vehicle"

    label = f"{label} {confidence:.2f}"

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )


# ============================================================
# CHECK PARKING SLOT OCCUPANCY
# ============================================================

occupied_count = 0
vacant_count = 0


for index, slot in enumerate(slots):

    # Convert slot points to NumPy array
    polygon = np.array(
        slot,
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Create mask for parking slot
    # --------------------------------------------------------

    slot_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.fillPoly(
        slot_mask,
        [polygon],
        255
    )

    slot_area = cv2.countNonZero(slot_mask)

    if slot_area == 0:
        vacant_count += 1
        continue


    # --------------------------------------------------------
    # Check each detected vehicle
    # --------------------------------------------------------

    occupied = False

    for x1, y1, x2, y2, class_id, confidence in vehicle_boxes:

        # Make sure coordinates are inside image
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue


        # ----------------------------------------------------
        # Vehicle mask
        # ----------------------------------------------------

        vehicle_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        cv2.rectangle(
            vehicle_mask,
            (x1, y1),
            (x2, y2),
            255,
            -1
        )


        # ----------------------------------------------------
        # Find intersection between vehicle and slot
        # ----------------------------------------------------

        intersection = cv2.bitwise_and(
            slot_mask,
            vehicle_mask
        )

        intersection_area = cv2.countNonZero(
            intersection
        )


        # Percentage of slot covered by vehicle
        overlap_ratio = (
            intersection_area / slot_area
        )


        if overlap_ratio >= OCCUPANCY_THRESHOLD:

            occupied = True
            break


    # ========================================================
    # DRAW SLOT
    # ========================================================

    if occupied:

        occupied_count += 1

        # RED = occupied
        color = (0, 0, 255)

        status = "OCCUPIED"

    else:

        vacant_count += 1

        # GREEN = vacant
        color = (0, 255, 0)

        status = "VACANT"


    # Draw parking slot polygon
    cv2.polylines(
        image,
        [polygon],
        True,
        color,
        3
    )


    # --------------------------------------------------------
    # Slot number position
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(polygon)

    text_x = x
    text_y = max(y - 8, 20)


    # Slot label
    cv2.putText(
        image,
        f"Slot {index + 1}: {status}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )


# ============================================================
# DISPLAY OCCUPANCY INFORMATION
# ============================================================

cv2.putText(
    image,
    f"Occupied: {occupied_count}",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 0, 255),
    3
)

cv2.putText(
    image,
    f"Vacant: {vacant_count}",
    (30, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 0),
    3
)

cv2.putText(
    image,
    f"Total Slots: {len(slots)}",
    (30, 130),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)


# ============================================================
# SAVE OUTPUT
# ============================================================

cv2.imwrite(
    OUTPUT_PATH,
    image
)


# ============================================================
# FINAL RESULT
# ============================================================

print("--------------------------------")
print("Parking Occupancy Detection")
print("--------------------------------")
print("Total slots :", len(slots))
print("Occupied    :", occupied_count)
print("Vacant      :", vacant_count)
print("Output      :", OUTPUT_PATH)
print("--------------------------------")


# ============================================================
# SHOW RESULT
# ============================================================

cv2.imshow(
    "ParkVision - Parking Occupancy",
    image
)

print("Press any key on the image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()