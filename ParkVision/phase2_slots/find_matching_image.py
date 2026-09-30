import cv2
import os
import glob

# -------------------------------------------------
# FILES
# -------------------------------------------------

TARGET = "dataset/images/parking.jpg"
IMAGE_FOLDER = r"C:\Users\Akshaya\Downloads\pklot_50\images"

# -------------------------------------------------
# LOAD TARGET IMAGE
# -------------------------------------------------

target = cv2.imread(TARGET)

if target is None:
    print("ERROR: parking.jpg not found")
    exit()

target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------
# ORB FEATURE DETECTOR
# -------------------------------------------------

orb = cv2.ORB_create(nfeatures=3000)

kp1, des1 = orb.detectAndCompute(target_gray, None)

if des1 is None:
    print("Could not find features in parking.jpg")
    exit()

# -------------------------------------------------
# SEARCH IMAGES
# -------------------------------------------------

files = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))

print("--------------------------------")
print("Finding matching parking image")
print("--------------------------------")
print("Target:", TARGET)
print("Images found:", len(files))
print("--------------------------------")

results = []

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for file in files:

    image = cv2.imread(file)

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb.detectAndCompute(gray, None)

    if des2 is None:
        continue

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only good matches
    good_matches = [
        m for m in matches
        if m.distance < 60
    ]

    score = len(good_matches)

    results.append((score, file))

# -------------------------------------------------
# SORT RESULTS
# -------------------------------------------------

results.sort(reverse=True)

print()
print("TOP MATCHES")
print("--------------------------------")

for rank, (score, file) in enumerate(results[:10], start=1):

    print(
        f"{rank}. {os.path.basename(file)} "
        f"--> {score} good matches"
    )

print("--------------------------------")

if results:

    best_score, best_file = results[0]

    print()
    print("BEST MATCH:")
    print(os.path.basename(best_file))
    print("Good matches:", best_score)
    print()
    print("Full path:")
    print(best_file)

else:

    print("No matching images found.")