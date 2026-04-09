import cv2
import numpy as np
import os

INPUT_FOLDER = "dataset\hsv_file"
OUTPUT_FOLDER = "dataset\hsv_out"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def extract_blue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # VERY wide range (don’t miss ink)
    # lower_blue = np.array([90, 20, 40])
    # upper_blue = np.array([180, 255, 255])
    lower_blue = np.array([80, 47, 20])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    return mask

for file in os.listdir(INPUT_FOLDER):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(INPUT_FOLDER, file)
        img = cv2.imread(path)

        mask = extract_blue(img)

        kernel = np.ones((3, 3), np.uint8)

        # # 1. Remove tiny noise (light open)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        #
        # # 2. Connect strokes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        #
        # 3. Remove very small junk
        # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        #
        # clean = np.zeros_like(mask)
        #
        # for i in range(1, num_labels):
        #     area = stats[i, cv2.CC_STAT_AREA]
        #     if area > 10:  # VERY LOW threshold
        #         clean[labels == i] = 255
        #
        # mask = clean
        #Blue-only image
        blue_img = cv2.bitwise_and(img, img, mask=mask)

        name = os.path.splitext(file)[0]

        # Save outputs
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_mask.png"), mask)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_blue.png"), blue_img)

        print(f"Saved debug for {file}")
