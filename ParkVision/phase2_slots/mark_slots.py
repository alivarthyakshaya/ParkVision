import cv2
import json

IMAGE_PATH = "dataset/images/parking.jpg"
SAVE_PATH = "phase2_slots/slots.json"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: parking.jpg not found")
    exit()

display = image.copy()

slots = []
drawing = False
start_x = 0
start_y = 0

print("--------------------------------")
print("ParkVision Slot Marking")
print("--------------------------------")
print("Draw exactly 7 parking slots.")
print()
print("LEFT CLICK + DRAG = draw slot")
print("R = reset current slot")
print("S = save slots")
print("Q = quit")
print("--------------------------------")


def mouse_callback(event, x, y, flags, param):

    global drawing, start_x, start_y, display

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        start_x = x
        start_y = y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:

        display = image.copy()

        # Draw already completed slots
        for i, (x1, y1, x2, y2) in enumerate(slots):

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Slot {i + 1}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Current rectangle
        cv2.rectangle(
            display,
            (start_x, start_y),
            (x, y),
            (255, 0, 0),
            2
        )

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        x1 = min(start_x, x)
        y1 = min(start_y, y)
        x2 = max(start_x, x)
        y2 = max(start_y, y)

        # Ignore extremely small rectangles
        if (x2 - x1) > 5 and (y2 - y1) > 5:

            if len(slots) < 7:

                slots.append((x1, y1, x2, y2))

                print(
                    f"Slot {len(slots)}: "
                    f"({x1}, {y1}, {x2}, {y2})"
                )

        display = image.copy()

        # Redraw all slots
        for i, (sx1, sy1, sx2, sy2) in enumerate(slots):

            cv2.rectangle(
                display,
                (sx1, sy1),
                (sx2, sy2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Slot {i + 1}",
                (sx1, max(20, sy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


cv2.namedWindow("Mark Parking Slots")
cv2.setMouseCallback("Mark Parking Slots", mouse_callback)

while True:

    cv2.imshow("Mark Parking Slots", display)

    key = cv2.waitKey(1) & 0xFF

    # Reset
    if key == ord("r"):

        slots = []
        display = image.copy()

        print("Slots reset.")

    # Save
    elif key == ord("s"):

        if len(slots) != 7:

            print(
                f"ERROR: You marked {len(slots)} slots."
            )
            print("You must mark exactly 7.")

        else:

            data = {
                "slots": slots
            }

            with open(SAVE_PATH, "w") as f:
                json.dump(data, f, indent=4)

            cv2.imwrite(
                "phase2_slots/marked_slots.jpg",
                display
            )

            print("--------------------------------")
            print("7 slots saved successfully!")
            print("Saved:", SAVE_PATH)
            print("Preview: phase2_slots/marked_slots.jpg")
            print("--------------------------------")

            break

    # Quit
    elif key == ord("q"):

        print("Cancelled.")
        break


cv2.destroyAllWindows()