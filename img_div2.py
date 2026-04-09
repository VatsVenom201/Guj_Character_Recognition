# ==============================
# IMPORTS
# ==============================
import cv2
import numpy as np
import os

# ==============================
# CONFIG
# ==============================
FOLDER_1 = "dataset/1fullpageclean"
FOLDER_2 = "dataset/2fullpageclean"
OUTPUT_DIR = "dataset/class_wise_images2"

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

# ==============================
# LABEL DICT (KEEP YOUR FULL DICT)
# ==============================
label_dict = {
    0: 'અ', 1: 'આ', 2: 'ઇ', 3: 'ઈ', 4: 'ઉ', 5: 'ઊ', 6: 'ઋ', 7: 'એ', 8: 'ઐ', 9: 'ઓ', 10: 'ઔ', 11: 'અં',
    12: 'ક', 13: 'કા', 14: 'કિ', 15: 'કી', 16: 'કુ', 17: 'કૂ', 18: 'કે', 19: 'કૈ', 20: 'કો', 21: 'કૌ', 22: 'કં', 23: 'કઃ',
    24: 'ખ', 25: 'ખા', 26: 'ખિ', 27: 'ખી', 28: 'ખુ', 29: 'ખૂ', 30: 'ખે', 31: 'ખૈ', 32: 'ખો', 33: 'ખૌ', 34: 'ખં', 35: 'ખઃ',
    36: 'ગ', 37: 'ગા', 38: 'ગિ', 39: 'ગી', 40: 'ગુ', 41: 'ગૂ', 42: 'ગે', 43: 'ગૈ', 44: 'ગો', 45: 'ગૌ', 46: 'ગં', 47: 'ગઃ',
    48: 'ઘ', 49: 'ઘા', 50: 'ઘિ', 51: 'ઘી', 52: 'ઘુ', 53: 'ઘૂ', 54: 'ઘે', 55: 'ઘૈ', 56: 'ઘો', 57: 'ઘૌ', 58: 'ઘં', 59: 'ઘઃ',
    60: 'ચ', 61: 'ચા', 62: 'ચિ', 63: 'ચી', 64: 'ચુ', 65: 'ચૂ', 66: 'ચે', 67: 'ચૈ', 68: 'ચો', 69: 'ચૌ', 70: 'ચં', 71: 'ચઃ',
    72: 'છ', 73: 'છા', 74: 'છિ', 75: 'છી', 76: 'છુ', 77: 'છૂ', 78: 'છે', 79: 'છૈ', 80: 'છો', 81: 'છૌ', 82: 'છં', 83: 'છઃ',
    84: 'જ', 85: 'જા', 86: 'જિ', 87: 'જી', 88: 'જુ', 89: 'જૂ', 90: 'જે', 91: 'જૈ', 92: 'જો', 93: 'જૌ', 94: 'જં', 95: 'જઃ',
    96: 'ઝ', 97: 'ઝા', 98: 'ઝિ', 99: 'ઝી', 100: 'ઝુ', 101: 'ઝૂ', 102: 'ઝે', 103: 'ઝૈ', 104: 'ઝો', 105: 'ઝૌ', 106: 'ઝં', 107: 'ઝઃ',
    108: 'ટ', 109: 'ટા', 110: 'ટિ', 111: 'ટી', 112: 'ટુ', 113: 'ટૂ', 114: 'ટે', 115: 'ટૈ', 116: 'ટો', 117: 'ટૌ', 118: 'ટં', 119: 'ટઃ',
    120: 'ઠ', 121: 'ઠા', 122: 'ઠિ', 123: 'ઠી', 124: 'ઠુ', 125: 'ઠૂ', 126: 'ઠે', 127: 'ઠૈ', 128: 'ઠો', 129: 'ઠૌ', 130: 'ઠં', 131: 'ઠઃ',
    132: 'ડ', 133: 'ડા', 134: 'ડિ', 135: 'ડી', 136: 'ડુ', 137: 'ડૂ', 138: 'ડે', 139: 'ડૈ', 140: 'ડો', 141: 'ડૌ', 142: 'ડં', 143: 'ડઃ',
    144: 'ઢ', 145: 'ઢા', 146: 'ઢિ', 147: 'ઢી', 148: 'ઢુ', 149: 'ઢૂ', 150: 'ઢે', 151: 'ઢૈ', 152: 'ઢો', 153: 'ઢૌ', 154: 'ઢં', 155: 'ઢઃ',
    156: 'ણ', 157: 'ણા', 158: 'ણિ', 159: 'ણી', 160: 'ણુ', 161: 'ણૂ', 162: 'ણે', 163: 'ણૈ', 164: 'ણો', 165: 'ણૌ', 166: 'ણં', 167: 'ણઃ',
    168: 'ત', 169: 'તા', 170: 'તિ', 171: 'તી', 172: 'તુ', 173: 'તૂ', 174: 'તે', 175: 'તૈ', 176: 'તો', 177: 'તૌ', 178: 'તં', 179: 'તઃ',
    180: 'થ', 181: 'થા', 182: 'થિ', 183: 'થી', 184: 'થુ', 185: 'થૂ', 186: 'થે', 187: 'થૈ', 188: 'થો', 189: 'થૌ', 190: 'થં', 191: 'થઃ',
    192: 'દ', 193: 'દા', 194: 'દિ', 195: 'દી', 196: 'દુ', 197: 'દૂ', 198: 'દે', 199: 'દૈ', 200: 'દો', 201: 'દૌ', 202: 'દં', 203: 'દઃ',
    204: 'ધ', 205: 'ધા', 206: 'ધિ', 207: 'ધી', 208: 'ધુ', 209: 'ધૂ', 210: 'ધે', 211: 'ધૈ', 212: 'ધો', 213: 'ધૌ', 214: 'ધં', 215: 'ધઃ',
    216: 'ન', 217: 'ના', 218: 'નિ', 219: 'ની', 220: 'નુ', 221: 'નૂ', 222: 'ને', 223: 'નૈ', 224: 'નો', 225: 'નૌ', 226: 'નં', 227: 'નઃ',
    228: 'પ', 229: 'પા', 230: 'પિ', 231: 'પી', 232: 'પુ', 233: 'પૂ', 234: 'પે', 235: 'પૈ', 236: 'પો', 237: 'પૌ', 238: 'પં', 239: 'પઃ',
    240: 'ફ', 241: 'ફા', 242: 'ફિ', 243: 'ફી', 244: 'ફુ', 245: 'ફૂ', 246: 'ફે', 247: 'ફૈ', 248: 'ફો', 249: 'ફૌ', 250: 'ફં', 251: 'ફઃ',
    252: 'બ', 253: 'બા', 254: 'બિ', 255: 'બી', 256: 'બુ', 257: 'બૂ', 258: 'બે', 259: 'બૈ', 260: 'બો', 261: 'બૌ', 262: 'બં', 263: 'બઃ',
    264: 'ભ', 265: 'ભા', 266: 'ભિ', 267: 'ભી', 268: 'ભુ', 269: 'ભૂ', 270: 'ભે', 271: 'ભૈ', 272: 'ભો', 273: 'ભૌ', 274: 'ભં', 275: 'ભઃ',
    276: 'મ', 277: 'મા', 278: 'મિ', 279: 'મી', 280: 'મુ', 281: 'મૂ', 282: 'મે', 283: 'મૈ', 284: 'મો', 285: 'મૌ', 286: 'મં', 287: 'મઃ',
    288: 'ય', 289: 'યા', 290: 'યિ', 291: 'યી', 292: 'યુ', 293: 'યૂ', 294: 'યે', 295: 'યૈ', 296: 'યો', 297: 'યૌ', 298: 'યં', 299: 'યઃ',
    300: 'ર', 301: 'રા', 302: 'રિ', 303: 'રી', 304: 'રુ', 305: 'રૂ', 306: 'રે', 307: 'રૈ', 308: 'રો', 309: 'રૌ', 310: 'રં', 311: 'રઃ',
    312: 'લ', 313: 'લા', 314: 'લિ', 315: 'લી', 316: 'લુ', 317: 'લૂ', 318: 'લે', 319: 'લૈ', 320: 'લો', 321: 'લૌ', 322: 'લં', 323: 'લઃ',
    324: 'વ', 325: 'વા', 326: 'વિ', 327: 'વી', 328: 'વુ', 329: 'વૂ', 330: 'વે', 331: 'વૈ', 332: 'વો', 333: 'વૌ', 334: 'વં', 335: 'વઃ',
    336: 'શ', 337: 'શા', 338: 'શિ', 339: 'શી', 340: 'શુ', 341: 'શૂ', 342: 'શે', 343: 'શૈ', 344: 'શો', 345: 'શૌ', 346: 'શં', 347: 'શઃ',
    348: 'ષ', 349: 'ષા', 350: 'ષિ', 351: 'ષી', 352: 'ષુ', 353: 'ષૂ', 354: 'ષે', 355: 'ષૈ', 356: 'ષો', 357: 'ષૌ', 358: 'ષં', 359: 'ષઃ',
    360: 'સ', 361: 'સા', 362: 'સિ', 363: 'સી', 364: 'સુ', 365: 'સૂ', 366: 'સે', 367: 'સૈ', 368: 'સો', 369: 'સૌ', 370: 'સં', 371: 'સઃ',
    372: 'હ', 373: 'હા', 374: 'હિ', 375: 'હી', 376: 'હુ', 377: 'હૂ', 378: 'હે', 379: 'હૈ', 380: 'હો', 381: 'હૌ', 382: 'હં', 383: 'હઃ',
    384: 'ળ', 385: 'ળા', 386: 'ળિ', 387: 'ળી', 388: 'ળુ', 389: 'ળૂ', 390: 'ળે', 391: 'ળૈ', 392: 'ળો', 393: 'ળૌ', 394: 'ળં', 395: 'ળઃ',
    396: 'ક્ષ', 397: 'ક્ષા', 398: 'ક્ષિ', 399: 'ક્ષી', 400: 'ક્ષુ', 401: 'ક્ષૂ', 402: 'ક્ષે', 403: 'ક્ષૈ', 404: 'ક્ષો', 405: 'ક્ષૌ', 406: 'ક્ષં', 407: 'ક્ષઃ',
    408: 'ત્ર', 409: 'ત્રા', 410: 'ત્રિ', 411: 'ત્રી', 412: 'ત્રુ', 413: 'ત્રૂ', 414: 'ત્રે', 415: 'ત્રૈ', 416: 'ત્રો', 417: 'ત્રૌ', 418: 'ત્રં', 419: 'ત્રઃ',
    420: 'જ્ઞ', 421: 'જ્ઞા', 422: 'જ્ઞિ', 423: 'જ્ઞી', 424: 'જ્ઞુ', 425: 'જ્ઞૂ', 426: 'જ્ઞે', 427: 'જ્ઞૈ', 428: 'જ્ઞો', 429: 'જ્ઞૌ', 430: 'જ્ઞં', 431: 'જ્ઞઃ'
}  # keep your full dict here


# ==============================
# CREATE OUTPUT FOLDERS
# ==============================
for i in range(len(label_dict)):
    os.makedirs(os.path.join(OUTPUT_DIR, f"{i:03d}"), exist_ok=True)

# ==============================
# FUNCTIONS
# ==============================
def cluster_points(points, axis=0, thresh=10):
    points = sorted(points, key=lambda x: x[axis])
    clusters = []
    current = [points[0]]

    for p in points[1:]:
        if abs(p[axis] - current[-1][axis]) < thresh:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]

    clusters.append(current)

    centers = [int(np.mean([p[axis] for p in cluster])) for cluster in clusters]
    return centers


# def process_folder(folder_path, start_index):
#     all_images = sorted([
#         f for f in os.listdir(folder_path)
#         if f.lower().endswith(valid_ext)
#     ])
#
#     for img_name in all_images:
#
#         print(f"\nProcessing: {img_name}")
#
#         INPUT_IMAGE = os.path.join(folder_path, img_name)
#
#         img = cv2.imread(INPUT_IMAGE)
#         if img is None:
#             print("❌ Failed to load image")
#             continue
#
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # ==============================
#         # THRESHOLD
#         # ==============================
#         thresh = cv2.adaptiveThreshold(
#             gray, 255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV,
#             15, 5
#         )
#
#         thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
#
#         # ==============================
#         # GRID DETECTION
#         # ==============================
#         h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
#         horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
#
#         v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
#         vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
#
#         intersections = cv2.bitwise_and(horizontal, vertical)
#
#         ys, xs = np.where(intersections > 0)
#         points = list(zip(xs, ys))
#
#         if len(points) == 0:
#             print("⚠️ No grid found, skipping")
#             continue
#
#         x_coords = sorted(cluster_points(points, axis=0))
#         y_coords = sorted(cluster_points(points, axis=1))
#
#         rows = len(y_coords) - 1
#         cols = len(x_coords) - 1
#
#         print(f"Grid: {rows} rows × {cols} cols")
#
#         index = start_index
#
#         # ==============================
#         # CELL EXTRACTION
#         # ==============================
#         for i in range(rows):
#             for j in range(cols):
#
#                 if index >= len(label_dict):
#                     print("✅ Completed all labels")
#                     return index
#
#                 x1, x2 = x_coords[j], x_coords[j+1]
#                 y1, y2 = y_coords[i], y_coords[i+1]
#
#                 margin = 2
#
#                 cell = img[
#                     y1 + margin: y2 - margin,
#                     x1 + margin: x2 - margin
#                 ]
#
#                 if cell is None or cell.size == 0:
#                     continue
#
#                 cell = cv2.resize(cell, (224, 224))
#
#                 folder = os.path.join(OUTPUT_DIR, f"{index:03d}")
#
#                 filename = os.path.join(
#                     folder,
#                     f"{img_name}_{len(os.listdir(folder))}.png"
#                 )
#
#                 cv2.imwrite(filename, cell)
#
#                 index += 1
#
#         start_index = index
#
#     return start_index
def has_text(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Count white pixels (text)
    white_pixels = np.sum(th == 255)
    total_pixels = th.size

    ratio = white_pixels / total_pixels

    # 🔥 tune this threshold
    return ratio > 0.02   # 2% pixels = text
def process_images(folder_path, start_index, end_index):
    all_images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_ext)
    ]

    all_images = sorted(
        all_images,
        key=lambda x: os.path.getmtime(os.path.join(folder_path, x))
    )

    index = start_index

    for img_name in all_images:

        print(f"\nProcessing: {img_name}")

        INPUT_IMAGE = os.path.join(folder_path, img_name)
        img = cv2.imread(INPUT_IMAGE)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 5
        )

        thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)

        # GRID
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
        vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

        intersections = cv2.bitwise_and(horizontal, vertical)

        ys, xs = np.where(intersections > 0)
        points = list(zip(xs, ys))

        if len(points) == 0:
            continue

        x_coords = sorted(cluster_points(points, axis=0))
        y_coords = sorted(cluster_points(points, axis=1))

        rows = len(y_coords) - 1
        cols = len(x_coords) - 1
        # ==============================
        # DEBUG VISUALIZATION (CORRECT PLACE)
        # ==============================
        debug_img = img.copy()

        # Draw grid lines
        for x in x_coords:
            cv2.line(debug_img, (x, 0), (x, debug_img.shape[0]), (0, 255, 0), 2)

        for y in y_coords:
            cv2.line(debug_img, (0, y), (debug_img.shape[1], y), (255, 0, 0), 2)

        # Draw LOCAL indices (ALWAYS START FROM 0)
        cell_index = 0

        for i in range(len(y_coords) - 1):
            for j in range(len(x_coords) - 1):
                x1, x2 = x_coords[j], x_coords[j + 1]
                y1, y2 = y_coords[i], y_coords[i + 1]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.putText(
                    debug_img,
                    str(cell_index),
                    (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

                cell_index += 1

        # Save ONCE per image
        os.makedirs("debug_outputs", exist_ok=True)
        cv2.imwrite(os.path.join("debug_outputs", f"debug_{img_name}"), debug_img)
        print(f"Saving label {index} from {img_name}")

        for i in range(rows):
            for j in range(cols):

                if index > end_index:
                    print("✅ Reached label limit")
                    return index

                x1, x2 = x_coords[j], x_coords[j + 1]
                y1, y2 = y_coords[i], y_coords[i + 1]

                cell = img[y1+2:y2-2, x1+2:x2-2]

                if cell is None or cell.size == 0:
                    continue
                if not has_text(cell):
                    continue

                cell = cv2.resize(cell, (224, 224))

                folder = os.path.join(OUTPUT_DIR, f"{index:03d}")

                filename = os.path.join(
                    folder,
                    f"{img_name}_{len(os.listdir(folder))}.png"
                )

                cv2.imwrite(filename, cell)

                index += 1


    return index
def process_folder1():
    print("🚀 Processing Folder 1 (0–215)")

    return process_images(FOLDER_1, start_index=0, end_index=215)
def process_folder2():
    print("🚀 Processing Folder 2 (216–431)")

    return process_images(FOLDER_2, start_index=216,end_index= 431)

# ==============================
# MAIN EXECUTION
# ==============================
print("🚀 STARTING PIPELINE...")

process_folder1()
process_folder2()

print("🎉 DONE")
