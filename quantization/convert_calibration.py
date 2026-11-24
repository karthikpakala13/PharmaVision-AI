import os
import cv2

SOURCE = "calibration_images"
DEST = "calibration_images_512"
SIZE = 512
PAD_COLOR = (114, 114, 114)

os.makedirs(DEST, exist_ok=True)

files = [
    f for f in os.listdir(SOURCE)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

converted = 0

for filename in files:
    path = os.path.join(SOURCE, filename)
    image = cv2.imread(path)

    if image is None:
        print("Skipping unreadable:", filename)
        continue

    h, w = image.shape[:2]

    scale = min(SIZE / w, SIZE / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image, (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = image = cv2.copyMakeBorder(
        resized,
        (SIZE - new_h) // 2,
        SIZE - new_h - (SIZE - new_h) // 2,
        (SIZE - new_w) // 2,
        SIZE - new_w - (SIZE - new_w) // 2,
        cv2.BORDER_CONSTANT,
        value=PAD_COLOR
    )

    output_path = os.path.join(DEST, filename)
    cv2.imwrite(output_path, canvas)
    converted += 1

print("Conversion complete.")
print("Images converted:", converted)
print("Output folder:", DEST)
