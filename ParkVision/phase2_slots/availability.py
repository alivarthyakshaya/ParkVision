import json

DATA_PATH = "phase2_slots/occupancy_data.json"

# --------------------------------
# LOAD OCCUPANCY DATA
# --------------------------------

with open(DATA_PATH, "r") as file:
    data = json.load(file)

total_slots = data["total_slots"]
occupied_count = data["occupied"]
vacant_count = data["vacant"]
slot_status = data["slot_status"]

# --------------------------------
# FIND AVAILABLE SLOTS
# --------------------------------

available_slots = []

for i, status in enumerate(slot_status):

    if status == "VACANT":
        available_slots.append(i + 1)

# --------------------------------
# DISPLAY
# --------------------------------

print()
print("================================")
print("          PARKVISION")
print("     PARKING AVAILABILITY")
print("================================")
print()

print("Total Slots     :", total_slots)
print("Occupied        :", occupied_count)
print("Available       :", vacant_count)

print()
print("Available Slots :")

if available_slots:

    for slot in available_slots:
        print("  Slot", slot)

else:
    print("  No parking slots available")

print()
print("================================")