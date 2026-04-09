import cv2
import numpy as np
import os

# ====== CONFIG ======
INPUT_FOLDER = "dataset\hsv_file"
OUTPUT_SUFFIX = "_ink.png"

# Pick first image from folder
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def skeletonize_opencv(img):
    img = img.copy()
    img[img != 0] = 255

    skel = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel

if not image_files:
    print("No images found!")
    exit()

image_path = os.path.join(INPUT_FOLDER, image_files[0])
img = cv2.imread(image_path)

# ====== STEP 1: Convert to LAB ======
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# ====== STEP 2: Detect blue ink using B channel ======
# Blue ink → lower B values (IMPORTANT)
_, mask_lab = cv2.threshold(B, 122, 255, cv2.THRESH_BINARY_INV)

# ====== STEP 3: Remove bright paper (intensity filter) ======
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask_dark = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

# Combine both
mask2 = cv2.bitwise_and(B,B, mask=mask_lab)
mask = cv2.bitwise_and(mask_lab, mask_dark)
cv2.imwrite("mask_lab.png", mask2)
cv2.imwrite("debug_B.png", B)
print("B min:", np.min(B), "B max:", np.max(B))
# ====== STEP 4: Clean noise ======
kernel = np.ones((2, 2), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# ====== STEP 5: (OPTIONAL) Connect broken strokes ======
# Uncomment ONLY if characters are breaking
# mask = cv2.dilate(mask, kernel, iterations=1)

# ====== STEP 6: Skeletonization ======
skeleton = skeletonize_opencv(mask)

# ====== STEP 7: Recover slight thickness ======
kernel = np.ones((2,2), np.uint8)
skeleton = cv2.dilate(skeleton, kernel, iterations=1)

# ====== STEP 8: Final smoothing ======
final = cv2.medianBlur(skeleton, 3)

# ====== DEBUG ======
cv2.imwrite("debug_mask.png", mask)

# ====== SAVE OUTPUT ======
output_path = os.path.join(INPUT_FOLDER, image_files[0].split('.')[0] + OUTPUT_SUFFIX)
cv2.imwrite(output_path, final)

print(f"Processed: {image_files[0]}")
print(f"Saved at: {output_path}")