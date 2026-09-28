from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Vehicle classes in COCO
vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Input image
image_path = "dataset/images/parking.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Run detection
results = model(image)

# Copy original image for drawing
output = image.copy()

# Vehicle counters
vehicle_count = {
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0
}

# Process detections
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    # Keep only vehicles
    if class_id in vehicle_classes:
        vehicle_name = vehicle_classes[class_id]

        vehicle_count[vehicle_name] += 1

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{vehicle_name} {confidence:.2f}"

        cv2.putText(
            output,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# Total vehicles
total = sum(vehicle_count.values())

# Display count
cv2.putText(
    output,
    f"Total Vehicles: {total}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Save result
cv2.imwrite("outputs/vehicle_detection.jpg", output)

# Print results
print("\n--- ParkVision Vehicle Detection ---")
print(f"Cars       : {vehicle_count['car']}")
print(f"Motorcycles: {vehicle_count['motorcycle']}")
print(f"Buses      : {vehicle_count['bus']}")
print(f"Trucks     : {vehicle_count['truck']}")
print(f"Total      : {total}")
print("------------------------------------")
print("Output saved to outputs/vehicle_detection.jpg")

# Show result
cv2.imshow("ParkVision - Vehicle Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()