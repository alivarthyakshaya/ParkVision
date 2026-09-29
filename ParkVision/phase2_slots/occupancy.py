import cv2
import os
from ultralytics import YOLO

# ============================================================
# PARKVISION - PARKING OCCUPANCY DETECTION
# Image: PKLot 640x640
# ============================================================

IMAGE_PATH = "dataset/images/parking.jpg"
OUTPUT_PATH = "phase2_slots/occupancy_result.jpg"

print("--------------------------------")
print("ParkVision Parking Occupancy")
print("--------------------------------")
print("Image:", os.path.abspath(IMAGE_PATH))

# ------------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: parking.jpg not found!")
    exit()

print("Image size:", image.shape[1], "x", image.shape[0])

# ------------------------------------------------------------
# 7 PARKING SLOTS
#
# These coordinates are for the CURRENT 640x640 image.
#
# Each tuple:
# (x1, y1, x2, y2)
# ------------------------------------------------------------

slots = [
    (433, 250, 456, 321),   # Slot 1
    (457, 250, 480, 321),   # Slot 2
    (481, 250, 504, 321),   # Slot 3
    (505, 250, 528, 321),   # Slot 4
    (529, 250, 552, 321),   # Slot 5
    (553, 250, 576, 321),   # Slot 6
    (577, 250, 600, 321)    # Slot 7
]

print("Total slots:", len(slots))

# ------------------------------------------------------------
# LOAD YOLO
# ------------------------------------------------------------

print("--------------------------------")
print("Loading YOLOv8s...")
print("--------------------------------")

model = YOLO("yolov8s.pt")

# Vehicle classes in COCO
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

VEHICLE_CLASSES = [2, 3, 5, 7]

occupied = 0

# ------------------------------------------------------------
# CHECK EACH SLOT
# ------------------------------------------------------------

for i, (x1, y1, x2, y2) in enumerate(slots):

    print("--------------------------------")
    print("Checking Slot", i + 1)
    print("--------------------------------")

    # Make sure coordinates are inside image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    # --------------------------------------------------------
    # Add padding around slot
    # --------------------------------------------------------

    pad = 12

    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(image.shape[1], x2 + pad)
    cy2 = min(image.shape[0], y2 + pad)

    crop = image[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        print("Invalid slot")
        continue

    # --------------------------------------------------------
    # ENLARGE SMALL PARKING SLOT
    # --------------------------------------------------------

    enlarged = cv2.resize(
        crop,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # YOLO DETECTION ON INDIVIDUAL SLOT
    # --------------------------------------------------------

    results = model.predict(
        enlarged,
        imgsz=640,
        conf=0.05,
        iou=0.45,
        classes=VEHICLE_CLASSES,
        verbose=False
    )

    best_conf = 0.0
    vehicle_found = False

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            conf = float(box.conf[0])

            if conf > best_conf:
                best_conf = conf

            if conf >= 0.08:
                vehicle_found = True

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if vehicle_found:

        status = "OCCUPIED"
        color = (0, 0, 255)
        occupied += 1

    else:

        status = "VACANT"
        color = (0, 255, 0)

    print(
        f"Slot {i + 1}: {status} "
        f"(confidence={best_conf:.3f})"
    )

    # --------------------------------------------------------
    # DRAW SLOT
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    # Label position
    label_y = max(20, y1 - 8)

    cv2.putText(
        image,
        f"Slot {i + 1}: {status}",
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )

# ------------------------------------------------------------
# FINAL COUNTS
# ------------------------------------------------------------

total = len(slots)
vacant = total - occupied

print()
print("================================")
print("FINAL RESULT")
print("================================")
print("Total slots :", total)
print("Occupied    :", occupied)
print("Vacant      :", vacant)
print("================================")

# ------------------------------------------------------------
# DISPLAY COUNTER
# ------------------------------------------------------------

cv2.putText(
    image,
    f"Occupied: {occupied}",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2
)

cv2.putText(
    image,
    f"Vacant: {vacant}",
    (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

cv2.putText(
    image,
    f"Total Slots: {total}",
    (20, 105),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

cv2.imwrite(OUTPUT_PATH, image)

print("Output:", os.path.abspath(OUTPUT_PATH))
print("================================")

# ------------------------------------------------------------
# SHOW
# ------------------------------------------------------------

cv2.imshow("ParkVision - Parking Occupancy", image)

print("Press any key on the image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()