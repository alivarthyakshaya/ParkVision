import cv2
import json
import os

# Parking image
IMAGE_PATH = "dataset/images/parking.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: parking image not found!")
    exit()

original = image.copy()

slots = []
current_points = []


def mouse_callback(event, x, y, flags, param):
    global current_points, image

    if event == cv2.EVENT_LBUTTONDOWN:

        current_points.append([x, y])

        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # Draw temporary lines
        if len(current_points) > 1:
            p1 = tuple(current_points[-2])
            p2 = tuple(current_points[-1])
            cv2.line(image, p1, p2, (0, 255, 0), 2)

        # Four points = one parking slot
        if len(current_points) == 4:

            pts = current_points + [current_points[0]]

            for i in range(len(pts) - 1):
                cv2.line(
                    image,
                    tuple(pts[i]),
                    tuple(pts[i + 1]),
                    (0, 255, 0),
                    2
                )

            # Slot number
            center_x = sum(p[0] for p in current_points) // 4
            center_y = sum(p[1] for p in current_points) // 4

            slot_number = len(slots) + 1

            cv2.putText(
                image,
                f"Slot {slot_number}",
                (center_x - 30, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            slots.append(current_points.copy())

            print(f"Slot {slot_number} saved")

            current_points = []


cv2.namedWindow("Parking Slot Marker")
cv2.setMouseCallback("Parking Slot Marker", mouse_callback)

print("\n--- Parking Slot Marker ---")
print("Click 4 corners of each parking slot.")
print("Press R to reset the current slot.")
print("Press S to save all slots.")
print("Press Q to quit.")
print("---------------------------")

while True:

    cv2.imshow("Parking Slot Marker", image)

    key = cv2.waitKey(1) & 0xFF

    # Reset current slot
    if key == ord("r"):
        current_points = []
        image = original.copy()

        # Redraw saved slots
        for i, slot in enumerate(slots):

            pts = slot + [slot[0]]

            for j in range(len(pts) - 1):
                cv2.line(
                    image,
                    tuple(pts[j]),
                    tuple(pts[j + 1]),
                    (0, 255, 0),
                    2
                )

            center_x = sum(p[0] for p in slot) // 4
            center_y = sum(p[1] for p in slot) // 4

            cv2.putText(
                image,
                f"Slot {i + 1}",
                (center_x - 30, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # Save slots
    elif key == ord("s"):

        os.makedirs("phase2_slots", exist_ok=True)

        with open("phase2_slots/slots.json", "w") as f:
            json.dump(slots, f, indent=4)

        cv2.imwrite(
            "phase2_slots/marked_slots.jpg",
            image
        )

        print(f"\nSaved {len(slots)} parking slots!")
        print("File: phase2_slots/slots.json")

    # Quit
    elif key == ord("q"):
        break

cv2.destroyAllWindows()