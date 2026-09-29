import cv2
from ultralytics import YOLO
import os

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

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found!")
    exit()

print("Image loaded:", image.shape)

model = YOLO("yolov8s.pt")

results = model.predict(
    source=image,
    imgsz=1280,
    conf=0.10,
    classes=[2, 3, 5, 7],
    verbose=False
)

result = results[0]

output = image.copy()

count = 0

if result.boxes is not None:

    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        conf = float(box.conf[0])

        cls = int(box.cls[0])

        count += 1

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        cv2.putText(
            output,
            f"Vehicle {conf:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

print("--------------------------------")
print("VEHICLES DETECTED:", count)
print("--------------------------------")

output_path = os.path.join(
    BASE_DIR,
    "phase2_slots",
    "detection_test.jpg"
)

cv2.imwrite(
    output_path,
    output
)

print("Saved:", output_path)

cv2.imshow(
    "YOLO Detection Test",
    output
)

cv2.waitKey(0)
cv2.destroyAllWindows()