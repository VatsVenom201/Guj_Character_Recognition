import cv2
import os
import numpy as np

INPUT_DIR = "dataset/class_wise_images"
OUTPUT_DIR = "dataset/preprocessed_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# PREPROCESS FUNCTION
# ==============================

def preprocess(img):

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. VERY LIGHT blur (reduce thickness issue)
    gray = cv2.GaussianBlur(gray, (1,1), 0)

    # 3. Threshold
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4. 🔥 Thin strokes (KEY STEP)
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    thresh = cv2.bitwise_not(thresh)

    # 5. Back to 3-channel
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    return thresh


# ==============================
# CENTERING FUNCTION
# ==============================
def center_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ink (non-white)
    coords = np.column_stack(np.where(gray < 250))

    if len(coords) == 0:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]

    size = max(w, h)
    square = np.ones((size, size, 3), dtype=np.uint8) * 255

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    return square

def pad_image(img, pad_ratio=0.5):

    h, w = img.shape[:2]

    # amount of padding
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)

    # create new white canvas
    new_h = h + 2 * pad_h
    new_w = w + 2 * pad_w

    padded = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255

    # place original image in center
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = img

    return padded

# ==============================
# SHARP MASK
# ==============================
def sharpen(img):  # not used
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(img, -1, kernel)
# ==============================
# MAIN LOOP
# ==============================
for class_id in sorted(os.listdir(INPUT_DIR)):

    class_path = os.path.join(INPUT_DIR, class_id)
    save_class_path = os.path.join(OUTPUT_DIR, class_id)

    os.makedirs(save_class_path, exist_ok=True)

    for img_name in sorted(os.listdir(class_path)):

        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # preprocess

         # sharpen edges

        img = preprocess(img)

        img = pad_image(img, pad_ratio=0.5)

        img = cv2.resize(img, (224, 224))
        # --- Save ---
        save_path = os.path.join(save_class_path, img_name)
        cv2.imwrite(save_path, img)

print("✅ Post-preprocessing complete!")