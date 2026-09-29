import json
import os
import webbrowser

DATA_PATH = "phase2_slots/occupancy_data.json"
IMAGE_PATH = "phase2_slots/occupancy_result.jpg"
HTML_PATH = "phase2_slots/dashboard.html"

# --------------------------------
# LOAD OCCUPANCY DATA
# --------------------------------

with open(DATA_PATH, "r") as file:
    data = json.load(file)

total = data["total_slots"]
occupied = data["occupied"]
vacant = data["vacant"]
slot_status = data["slot_status"]

# Find vacant slots
available_slots = []

for i, status in enumerate(slot_status):
    if status == "VACANT":
        available_slots.append(i + 1)

# --------------------------------
# CREATE SLOT CARDS
# --------------------------------

slot_cards = ""

for i, status in enumerate(slot_status):

    slot_number = i + 1

    if status == "OCCUPIED":
        slot_cards += f"""
        <div class="slot occupied">
            <h3>Slot {slot_number}</h3>
            <p>OCCUPIED</p>
        </div>
        """
    else:
        slot_cards += f"""
        <div class="slot vacant">
            <h3>Slot {slot_number}</h3>
            <p>VACANT</p>
        </div>
        """

# --------------------------------
# AVAILABLE SLOTS TEXT
# --------------------------------

if available_slots:
    available_text = ", ".join(
        "Slot " + str(slot) for slot in available_slots
    )
else:
    available_text = "No slots available"

# --------------------------------
# CREATE DASHBOARD
# --------------------------------

html = f"""
<!DOCTYPE html>
<html>
<head>

<title>ParkVision Dashboard</title>

<style>

body {{
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f2f5f9;
}}

.header {{
    background: #1e293b;
    color: white;
    padding: 20px;
    text-align: center;
}}

.header h1 {{
    margin: 0;
    font-size: 32px;
}}

.container {{
    width: 90%;
    margin: 25px auto;
}}

.stats {{
    display: flex;
    gap: 20px;
    justify-content: center;
    margin-bottom: 25px;
}}

.card {{
    background: white;
    padding: 20px;
    width: 200px;
    text-align: center;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
}}

.card h2 {{
    margin: 5px;
    font-size: 32px;
}}

.total {{
    color: #2563eb;
}}

.red {{
    color: #dc2626;
}}

.green {{
    color: #16a34a;
}}

.image-box {{
    background: white;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
}}

.image-box img {{
    width: 100%;
    max-width: 1000px;
    border-radius: 8px;
}}

.slots {{
    display: flex;
    gap: 15px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 25px;
}}

.slot {{
    width: 150px;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
}}

.slot h3 {{
    margin: 5px;
}}

.slot p {{
    margin: 5px;
    font-weight: bold;
}}

.occupied {{
    background: #dc2626;
}}

.vacant {{
    background: #16a34a;
}}

.available {{
    background: white;
    padding: 20px;
    margin-top: 25px;
    text-align: center;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
}}

.available h2 {{
    color: #16a34a;
}}

</style>

</head>

<body>

<div class="header">
    <h1>🚗 ParkVision</h1>
    <p>Smart Parking Occupancy Dashboard</p>
</div>

<div class="container">

    <div class="stats">

        <div class="card">
            <h3>Total Slots</h3>
            <h2 class="total">{total}</h2>
        </div>

        <div class="card">
            <h3>Occupied</h3>
            <h2 class="red">{occupied}</h2>
        </div>

        <div class="card">
            <h3>Vacant</h3>
            <h2 class="green">{vacant}</h2>
        </div>

    </div>

    <div class="image-box">

        <h2>Parking Lot Analysis</h2>

        <img src="occupancy_result.jpg">

    </div>

    <div class="available">

        <h2>Available Parking Slots</h2>

        <h3>{available_text}</h3>

    </div>

    <div class="slots">

        {slot_cards}

    </div>

</div>

</body>
</html>
"""

# --------------------------------
# SAVE DASHBOARD
# --------------------------------

with open(HTML_PATH, "w", encoding="utf-8") as file:
    file.write(html)

print()
print("--------------------------------")
print("       PARKVISION DASHBOARD")
print("--------------------------------")
print()
print("Total Slots :", total)
print("Occupied    :", occupied)
print("Vacant      :", vacant)
print("Available   :", available_text)
print()
print("Dashboard created!")
print("Opening browser...")
print()

# Open dashboard
webbrowser.open(
    "file:///" + os.path.abspath(HTML_PATH)
)