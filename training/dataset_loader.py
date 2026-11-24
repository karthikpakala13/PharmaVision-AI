import os
import cv2
import numpy as np
import tensorflow as tf

DATASET_ROOT = "/home/karthik/arm2/datasets"
IMAGE_SIZE = 512
MAX_BOXES = 20

def load_sample(split, filename):
    folder = os.path.join(DATASET_ROOT, split)

    image_path = os.path.join(folder, filename)
    label_path = os.path.splitext(image_path)[0] + ".txt"

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Unexpected image size: {image_path}")

    image = image.astype(np.float32) / 255.0

    boxes = []

    if os.path.exists(label_path):
        with open(label_path, "r") as file:
            for line in file:
                values = line.split()

                if len(values) != 5:
                    raise ValueError(f"Invalid annotation: {label_path}")

                cls, cx, cy, w, h = map(float, values)

                if cls not in (0, 1):
                    raise ValueError(f"Invalid class ID: {cls}")

                if not (
                    0 <= cx <= 1 and
                    0 <= cy <= 1 and
                    0 < w <= 1 and
                    0 < h <= 1
                ):
                    raise ValueError(f"Invalid box coordinates: {line}")

                boxes.append([cls, cx, cy, w, h])

    if len(boxes) > MAX_BOXES:
        raise ValueError(
            f"Too many boxes in {filename}: {len(boxes)}"
        )

    padded_boxes = np.full((MAX_BOXES, 5), -1, dtype=np.float32)

    if boxes:
        padded_boxes[:len(boxes)] = np.array(boxes, dtype=np.float32)

    return image, padded_boxes


def create_dataset(split, batch_size=1, shuffle=False):
    folder = os.path.join(DATASET_ROOT, split)

    filenames = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if shuffle:
        rng = np.random.default_rng(42)
        rng.shuffle(filenames)

    def generator():
        for filename in filenames:
            yield load_sample(split, filename)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((512, 512, 3), tf.float32),
            tf.TensorSpec((MAX_BOXES, 5), tf.float32)
        )
    )

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(1)

    return dataset, len(filenames)


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        print(f"\nTesting {split} dataset...")

        dataset, count = create_dataset(
            split,
            batch_size=1,
            shuffle=False
        )

        images, boxes = next(iter(dataset))

        print("Total images:", count)
        print("Image batch shape:", images.shape)
        print("Box batch shape:", boxes.shape)
        print("Pixel range:",
              float(tf.reduce_min(images)),
              float(tf.reduce_max(images)))

        valid_boxes = tf.reduce_sum(
            tf.cast(boxes[0, :, 0] >= 0, tf.int32)
        )

        print("Boxes in first image:", int(valid_boxes))

    print("\nDataset loader test completed.")
