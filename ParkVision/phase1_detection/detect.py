from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Read parking image
image_path = "dataset/images/parking.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Run YOLO detection
results = model(image)

# Get annotated image
output = results[0].plot()

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Save result
cv2.imwrite("outputs/detected.jpg", output)

# Display result
cv2.imshow("ParkVision - Vehicle Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Detection completed!")
print("Output saved to outputs/detected.jpg")