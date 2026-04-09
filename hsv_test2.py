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

# ====== STEP 1: Convert to HSV ======
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Blue ink range (tweak if needed)

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([140, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)

# ====== STEP 2: Clean noise (SAFE) ======
kernel = np.ones((2, 2), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# # Connect broken strokes (VERY LIGHT)
# kernel = np.ones((2,2), np.uint8)
# mask = cv2.dilate(mask, kernel, iterations=1)

# ====== STEP 3: (REMOVE EDGE ANDING) ❌ REMOVE THIS
# mask = cv2.bitwise_and(mask, edges)

# ====== STEP 4: Skeleton (fixed input) ======
skeleton = skeletonize_opencv(mask)

# ====== STEP 5: Light dilation to recover strokes ======
kernel = np.ones((2,2), np.uint8)
skeleton = cv2.dilate(skeleton, kernel, iterations=1)

# ====== STEP 6: Final smoothing ======
final = cv2.medianBlur(skeleton, 3)
cv2.imwrite("debug_mask.png", mask)
# ====== SAVE OUTPUT ======
output_path = os.path.join(INPUT_FOLDER, image_files[0].split('.')[0] + OUTPUT_SUFFIX)
cv2.imwrite(output_path, final)

print(f"Processed: {image_files[0]}")
print(f"Saved at: {output_path}")