import cv2
import json
from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------
IMAGE_PATH = "dataset/images/parking.jpg"
SLOTS_PATH = "phase2_slots/slots.json"
MODEL_PATH = "yolov8n.pt"

# -----------------------------
# Load model and image
# -----------------------------
model = YOLO(MODEL_PATH)
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: Image not found!")
    exit()

# -----------------------------
# Load parking slots
# -----------------------------
with open(SLOTS_PATH, "r") as f:
    slots = json.load(f)

# -----------------------------
# Detect vehicles
# -----------------------------
results = model(image, conf=0.25)

vehicles = []

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in ["car", "bus", "truck", "motorcycle"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Vehicle center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            vehicles.append((cx, cy))

# -----------------------------
# Check slot occupancy
# -----------------------------
occupied = 0

for i, slot in enumerate(slots):

    points = []
    for point in slot:
        points.append([int(point[0]), int(point[1])])

    polygon = __import__("numpy").array(points, dtype="int32")

    is_occupied = False

    for cx, cy in vehicles:

        inside = cv2.pointPolygonTest(
            polygon,
            (cx, cy),
            False
        )

        if inside >= 0:
            is_occupied = True
            break

    if is_occupied:
        occupied += 1
        color = (0, 0, 255)
        status = "OCCUPIED"
    else:
        color = (0, 255, 0)
        status = "VACANT"

    # Draw slot
    cv2.polylines(
        image,
        [polygon],
        True,
        color,
        3
    )

    # Slot label
    x, y = points[0]

    cv2.putText(
        image,
        f"Slot {i+1}: {status}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

# -----------------------------
# Display results
# -----------------------------
vacant = len(slots) - occupied

cv2.putText(
    image,
    f"Occupied: {occupied}",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    3
)

cv2.putText(
    image,
    f"Vacant: {vacant}",
    (30, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    3
)

print("--------------------------------")
print(f"Total Slots : {len(slots)}")
print(f"Occupied    : {occupied}")
print(f"Vacant      : {vacant}")
print("--------------------------------")

cv2.imwrite(
    "phase2_slots/occupancy_result.jpg",
    image
)
cv2.imshow("ParkVision - Parking Occupancy", image)
cv2.waitKey(0)
cv2.destroyAllWindows()