import numpy as np

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 512
NUM_CLASSES = 2
NUM_ANCHORS = 3
MAX_BOXES = 20

# Anchor order: 16x16, 32x32, 64x64
ANCHORS = [
    [[27.0, 77.0], [40.0, 53.5], [37.0, 78.0]],
    [[30.0, 46.0], [25.0, 58.0], [32.0, 64.0]],
    [[14.0, 27.0], [18.0, 32.0], [23.0, 38.0]]
]

GRID_SIZES = [16, 32, 64]


# --------------------------------------------------
# ANCHOR IOU
# --------------------------------------------------

def calculate_wh_iou(box_wh, anchors):
    """
    Calculate width-height IoU between one GT box
    and all anchors.

    box_wh: [width, height]
    anchors: [N, 2]
    """

    box_area = box_wh[0] * box_wh[1]
    anchor_area = anchors[:, 0] * anchors[:, 1]

    intersection_w = np.minimum(box_wh[0], anchors[:, 0])
    intersection_h = np.minimum(box_wh[1], anchors[:, 1])

    intersection = intersection_w * intersection_h

    union = box_area + anchor_area - intersection

    return intersection / np.maximum(union, 1e-9)


# --------------------------------------------------
# TARGET ASSIGNMENT
# --------------------------------------------------

def assign_targets(boxes):

    """
    Convert YOLO labels into YOLOv4 training targets.

    Input:
        boxes: [MAX_BOXES, 5]

        Each box:
        [class_id, cx, cy, width, height]

        Coordinates are normalized between 0 and 1.

    Output:
        targets:
            [
                [16,16,3,7],
                [32,32,3,7],
                [64,64,3,7]
            ]

        valid_box_count
        collision_count
    """

    # Initialize target tensors
    targets = [
        np.zeros((16, 16, 3, 7), dtype=np.float32),
        np.zeros((32, 32, 3, 7), dtype=np.float32),
        np.zeros((64, 64, 3, 7), dtype=np.float32)
    ]

    valid_boxes = []
    collisions = 0

    all_anchors = np.array(
        ANCHORS[2] + ANCHORS[1] + ANCHORS[0],
        dtype=np.float32
    )

    # --------------------------------------------------
    # FILTER AND VALIDATE GROUND-TRUTH BOXES
    # --------------------------------------------------

    for box in boxes:

        class_id, cx, cy, width, height = box

        # Ignore padded entries
        if class_id < 0:
            continue

        if not float(class_id).is_integer():
            raise ValueError(
                f"Invalid class ID: {class_id}"
            )

        class_id = int(class_id)

        if class_id not in range(NUM_CLASSES):
            raise ValueError(
                f"Invalid class ID: {class_id}"
            )

        if not np.all(np.isfinite(box)):
            raise ValueError(
                f"Non-finite bounding box: {box}"
            )

        if not (
            0.0 <= cx <= 1.0
            and 0.0 <= cy <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError(
                f"Invalid normalized bounding box: {box}"
            )

        valid_boxes.append([
            class_id,
            cx,
            cy,
            width,
            height
        ])

    # --------------------------------------------------
    # ASSIGN EACH BOX TO ITS BEST ANCHOR
    # --------------------------------------------------

    for box in valid_boxes:

        class_id, cx, cy, width, height = box

        # Convert normalized dimensions into pixels
        box_wh_pixels = np.array(
            [width * IMAGE_SIZE, height * IMAGE_SIZE],
            dtype=np.float32
        )

        # Calculate IoU against all nine anchors
        ious = calculate_wh_iou(
            box_wh_pixels,
            all_anchors
        )

        best_anchor_index = int(np.argmax(ious))

        # Map global anchor index to output scale
        if best_anchor_index < 3:
            scale_index = 2
            anchor_index = best_anchor_index

        elif best_anchor_index < 6:
            scale_index = 1
            anchor_index = best_anchor_index - 3

        else:
            scale_index = 0
            anchor_index = best_anchor_index - 6

        grid_size = GRID_SIZES[scale_index]

        anchor_w, anchor_h = ANCHORS[scale_index][anchor_index]

        # --------------------------------------------------
        # FIND GRID CELL
        # --------------------------------------------------

        grid_x = min(
            int(cx * grid_size),
            grid_size - 1
        )

        grid_y = min(
            int(cy * grid_size),
            grid_size - 1
        )

        # --------------------------------------------------
        # CHECK COLLISIONS
        # --------------------------------------------------

        target = targets[scale_index]

        if target[grid_y, grid_x, anchor_index, 4] == 1:

            collisions += 1

            continue

        # --------------------------------------------------
        # ENCODE BOX TARGETS
        # --------------------------------------------------

        # Fractional offsets inside the grid cell
        tx = cx * grid_size - grid_x
        ty = cy * grid_size - grid_y

        # Log-space width and height relative to anchor
        tw = np.log(
            (width * IMAGE_SIZE) / anchor_w
        )

        th = np.log(
            (height * IMAGE_SIZE) / anchor_h
        )

        # --------------------------------------------------
        # STORE TARGET
        # --------------------------------------------------

        target[grid_y, grid_x, anchor_index, 0] = tx
        target[grid_y, grid_x, anchor_index, 1] = ty

        target[grid_y, grid_x, anchor_index, 2] = tw
        target[grid_y, grid_x, anchor_index, 3] = th

        # Objectness
        target[grid_y, grid_x, anchor_index, 4] = 1.0

        # One-hot class encoding
        target[grid_y, grid_x, anchor_index, 5 + class_id] = 1.0

    return targets, len(valid_boxes), collisions


# --------------------------------------------------
# DATASET VALIDATION
# --------------------------------------------------

if __name__ == "__main__":

    from dataset_loader import load_sample, DATASET_ROOT
    import os

    print("\nYOLOv4 TARGET ASSIGNMENT VALIDATION")
    print("=" * 55)

    for split in ["train", "valid", "test"]:

        split_path = os.path.join(
            DATASET_ROOT,
            split
        )

        image_files = sorted([
            filename
            for filename in os.listdir(split_path)
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        total_gt = 0
        total_assigned = 0
        total_collisions = 0

        scale_counts = [0, 0, 0]

        for filename in image_files:

            _, boxes = load_sample(split, filename)

            targets, valid_count, collision_count = (
                assign_targets(boxes)
            )

            total_gt += valid_count
            total_collisions += collision_count

            assigned_count = sum(
                int(np.sum(target[..., 4]))
                for target in targets
            )

            total_assigned += assigned_count

            for i in range(3):

                scale_counts[i] += int(
                    np.sum(targets[i][..., 4])
                )

        print(f"\nSplit: {split}")
        print(f"Images: {len(image_files)}")
        print(f"Ground-truth boxes: {total_gt}")
        print(f"Assigned boxes: {total_assigned}")
        print(f"Collisions: {total_collisions}")

        print(
            f"16x16 assignments: {scale_counts[0]}"
        )

        print(
            f"32x32 assignments: {scale_counts[1]}"
        )

        print(
            f"64x64 assignments: {scale_counts[2]}"
        )

        if total_gt != total_assigned:
            raise RuntimeError(
                f"{split}: Some boxes were not assigned."
            )

        if total_collisions != 0:
            raise RuntimeError(
                f"{split}: Target assignment collisions detected."
            )

    print("\nTARGET ASSIGNMENT VALIDATION PASSED.")
