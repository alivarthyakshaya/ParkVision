import cv2
import json
import os


from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# ==========================================
# FILES
# ==========================================

IMAGE_PATH = "dataset/images/parking.jpg"
SLOTS_PATH = "phase2_slots/slots.json"
OUTPUT_PATH = "phase2_slots/occupancy_result.jpg"


# ==========================================
# SETTINGS
# ==========================================

CONFIDENCE = 0.25
IOU_MATCH_THRESHOLD = 0.05


# ==========================================
# LOAD IMAGE
# ==========================================

print("--------------------------------")
print("ParkVision Parking Occupancy")
print("--------------------------------")

print("Image:", IMAGE_PATH)

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: parking.jpg not found!")
    exit()

height, width = image.shape[:2]

print("Image size:", width, "x", height)


# ==========================================
# LOAD OUR 7 SELECTED SLOTS
# ==========================================

with open(SLOTS_PATH, "r") as f:
    data = json.load(f)

slots = data["slots"]

print("Total slots:", len(slots))


# ==========================================
# DOWNLOAD / LOAD PKLOT YOLO MODEL
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
# IOU FUNCTION
# ==========================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ==========================================
# RUN PKLOT MODEL
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
# GET ALL DETECTIONS
# ==========================================

detections = []

if result.boxes is not None:

    for box in result.boxes:

        coords = box.xyxy[0].tolist()

        x1, y1, x2, y2 = map(int, coords)

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        # PKLot:
        # 0 = vacant
        # 1 = occupied

        if class_id == 0:
            status = "VACANT"

        elif class_id == 1:
            status = "OCCUPIED"

        else:
            continue

        detections.append({
            "box": [x1, y1, x2, y2],
            "confidence": confidence,
            "status": status
        })


print("Parking spaces detected:", len(detections))


# ==========================================
# DRAW RESULTS
# ==========================================

output = image.copy()

occupied_count = 0
vacant_count = 0


# ==========================================
# MATCH OUR 7 SLOTS
# ==========================================

for number, slot in enumerate(slots, start=1):

    best_detection = None
    best_iou = 0.0

    # Find the PKLot detection
    # that best matches this slot

    for detection in detections:

        iou = calculate_iou(
            slot,
            detection["box"]
        )

        if iou > best_iou:

            best_iou = iou
            best_detection = detection


    # ======================================
    # DECIDE STATUS
    # ======================================

    if (
        best_detection is not None
        and best_iou >= IOU_MATCH_THRESHOLD
    ):

        status = best_detection["status"]
        confidence = best_detection["confidence"]

    else:

        status = "UNKNOWN"
        confidence = 0.0


    # ======================================
    # COUNT
    # ======================================

    if status == "OCCUPIED":

        occupied_count += 1
        color = (0, 0, 255)

    elif status == "VACANT":

        vacant_count += 1
        color = (0, 255, 0)

    else:

        color = (0, 255, 255)


    # ======================================
    # TERMINAL OUTPUT
    # ======================================

    print("--------------------------------")
    print("Checking Slot", number)
    print("--------------------------------")

    print(
        "Slot", number,
        ":", status,
        "| IoU =",
        round(best_iou, 3),
        "| confidence =",
        round(confidence, 3)
    )


    # ======================================
    # DRAW SLOT
    # ======================================

    x1, y1, x2, y2 = slot

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    cv2.putText(
        output,
        f"Slot {number}: {status}",
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2
    )


# ==========================================
# FINAL RESULT
# ==========================================

total = len(slots)

print()
print("================================")
print("FINAL RESULT")
print("================================")

print("Total slots :", total)
print("Occupied    :", occupied_count)
print("Vacant      :", vacant_count)

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
# SAVE
# ==========================================

os.makedirs("phase2_slots", exist_ok=True)

cv2.imwrite(
    OUTPUT_PATH,
    output
)

# ==========================================
# SAVE OCCUPANCY DATA
# ==========================================

occupancy_data = {
    "total_slots": total,
    "occupied": occupied_count,
    "vacant": vacant_count,
    "slot_status": []
}

# Re-check each slot to store its status
for slot in slots:

    best_detection = None
    best_iou = 0.0

    for detection in detections:

        iou = calculate_iou(
            slot,
            detection["box"]
        )

        if iou > best_iou:
            best_iou = iou
            best_detection = detection

    if (
        best_detection is not None
        and best_iou >= IOU_MATCH_THRESHOLD
    ):
        occupancy_data["slot_status"].append(
            best_detection["status"]
        )
    else:
        occupancy_data["slot_status"].append(
            "UNKNOWN"
        )


with open(
    "phase2_slots/occupancy_data.json",
    "w"
) as f:

    json.dump(
        occupancy_data,
        f,
        indent=4
    )

print("Occupancy data saved.")


print("Output:", OUTPUT_PATH)
print("================================")


# ==========================================
# DISPLAY
# ==========================================

cv2.imshow(
    "ParkVision Parking Occupancy",
    output
)

cv2.waitKey(0)
cv2.destroyAllWindows()