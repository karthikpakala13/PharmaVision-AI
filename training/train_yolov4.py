
import os
import numpy as np
import tensorflow as tf

from dataset_loader import create_dataset
from target_assignment import assign_targets
from yolov4_loss import yolov4_loss


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = "/home/karthik/arm2"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "yolov4_2class_custom.h5"
)

SMOKE_CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "head_only_smoke.weights.h5"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_DIR,
    "checkpoints"
)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_batch2.weights.h5"
)

LATEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "latest_batch2.weights.h5"
)

BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 1e-4

TRAINABLE_LAYERS = [
    "conv2d_93",
    "conv2d_101",
    "conv2d_109"
]


# ============================================================
# GPU CONFIGURATION
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

print("\nGPUs detected:", gpus)

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(
            gpu,
            True
        )
    except RuntimeError as error:
        print("GPU memory configuration:", error)


# ============================================================
# PREPARE GROUND-TRUTH BOXES
# ============================================================

def prepare_gt_boxes(boxes):
    """
    Dataset format:
        [class_id, cx, cy, width, height]

    Loss format:
        [valid_flag, cx, cy, width, height]

    Padding:
        class_id = -1
        valid_flag = -1
    """

    valid_flag = tf.where(
        boxes[..., 0] >= 0,
        tf.ones_like(boxes[..., 0]),
        -tf.ones_like(boxes[..., 0])
    )

    gt_boxes = tf.concat(
        [
            valid_flag[..., None],
            boxes[..., 1:5]
        ],
        axis=-1
    )

    return gt_boxes


# ============================================================
# CREATE BATCHED YOLO TARGETS
# ============================================================

def prepare_batch_targets(boxes):
    """
    Input:
        boxes: [B, 20, 5]

    Output:
        Three target tensors:
        [B, 16, 16, 3, 7]
        [B, 32, 32, 3, 7]
        [B, 64, 64, 3, 7]
    """

    batch_size = int(tf.shape(boxes)[0].numpy())

    batch_targets = [
        [],
        [],
        []
    ]

    for i in range(batch_size):

        sample_boxes = boxes[i].numpy()

        targets, valid_count, collision_count = (
            assign_targets(sample_boxes)
        )

        if collision_count != 0:
            raise RuntimeError(
                f"Target assignment collision in batch sample {i}: "
                f"{collision_count}"
            )

        # Confirm all valid ground-truth boxes were assigned.
        assigned_count = sum(
            int(np.sum(target[..., 4]))
            for target in targets
        )

        if assigned_count != valid_count:
            raise RuntimeError(
                f"Target assignment mismatch: "
                f"{assigned_count} assigned, "
                f"{valid_count} ground-truth boxes"
            )

        for scale_index in range(3):
            batch_targets[scale_index].append(
                targets[scale_index]
            )

    batch_targets = [
        tf.convert_to_tensor(
            np.stack(scale_targets, axis=0),
            dtype=tf.float32
        )
        for scale_targets in batch_targets
    ]

    return batch_targets


# ============================================================
# LOSS CALCULATION
# ============================================================

def calculate_loss(model, images, boxes, training):
    targets = prepare_batch_targets(boxes)

    gt_boxes = prepare_gt_boxes(boxes)

    predictions = model(
        images,
        training=training
    )

    loss_dict = yolov4_loss(
        predictions,
        targets,
        gt_boxes
    )

    return loss_dict


# ============================================================
# TRAINING STEP
# ============================================================

@tf.function
def train_step(images, targets, gt_boxes):

    with tf.GradientTape() as tape:

        predictions = model(
            images,
            training=True
        )

        loss_dict = yolov4_loss(
            predictions,
            targets,
            gt_boxes
        )

        total_loss = loss_dict["total"]

    gradients = tape.gradient(
        total_loss,
        model.trainable_variables
    )

    if any(g is None for g in gradients):
        raise RuntimeError(
            "A trainable variable has no gradient."
        )

    gradients, _ = tf.clip_by_global_norm(
        gradients,
        10.0
    )

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )

    return loss_dict


# ============================================================
# VALIDATION STEP
# ============================================================

def validation_step(images, boxes):

    targets = prepare_batch_targets(boxes)

    gt_boxes = prepare_gt_boxes(boxes)

    predictions = model(
        images,
        training=False
    )

    loss_dict = yolov4_loss(
        predictions,
        targets,
        gt_boxes
    )

    return loss_dict


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    print("\nLoading custom YOLOv4 model...")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("\nModel outputs:")

    for output in model.outputs:
        print(output.shape)

    # --------------------------------------------------------
    # FREEZE BACKBONE
    # --------------------------------------------------------

    for layer in model.layers:
        layer.trainable = False

    # Unfreeze only the three detection-head convolutions.
    for layer_name in TRAINABLE_LAYERS:

        layer = model.get_layer(layer_name)

        layer.trainable = True

        print("Trainable layer:", layer.name)

    print(
        "\nTrainable variables:",
        len(model.trainable_variables)
    )

    if len(model.trainable_variables) == 0:
        raise RuntimeError(
            "No trainable variables found."
        )

    # --------------------------------------------------------
    # RESUME FROM LATEST TRAINING CHECKPOINT
    # --------------------------------------------------------

    if not os.path.isfile(LATEST_CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {LATEST_CHECKPOINT}"
        )

    print("\nLoading latest training checkpoint...")

    model.load_weights(LATEST_CHECKPOINT)

    print("Latest training checkpoint loaded.")

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    train_dataset, train_count = create_dataset(
        "train",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("Training images:", train_count)
    print(
        "Training batches:",
        (train_count + BATCH_SIZE - 1) // BATCH_SIZE
    )

    print("\nLoading validation dataset...")

    valid_dataset, valid_count = create_dataset(
        "valid",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Validation images:", valid_count)
    print(
        "Validation batches:",
        (valid_count + BATCH_SIZE - 1) // BATCH_SIZE
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_loss = 2.966870

    print("\nStarting YOLOv4 training")
    print("=" * 65)

    for epoch in range(9, EPOCHS):

        print(f"\nEPOCH {epoch + 1}/{EPOCHS}")
        print("-" * 65)

        train_loss_sum = 0.0
        train_box_sum = 0.0
        train_obj_sum = 0.0
        train_cls_sum = 0.0

        train_batches = 0

        # ====================================================
        # TRAIN
        # ====================================================

        for step, (images, boxes) in enumerate(
            train_dataset
        ):

            targets = prepare_batch_targets(boxes)

            gt_boxes = prepare_gt_boxes(boxes)

            loss_dict = train_step(
                images,
                targets,
                gt_boxes
            )

            train_loss_sum += float(
                loss_dict["total"].numpy()
            )

            train_box_sum += float(
                loss_dict["box"].numpy()
            )

            train_obj_sum += float(
                loss_dict["objectness"].numpy()
            )

            train_cls_sum += float(
                loss_dict["classification"].numpy()
            )

            train_batches += 1

            if step % 50 == 0:

                print(
                    f"Step {step + 1}: "
                    f"Total={float(loss_dict['total'].numpy()):.4f}, "
                    f"Box={float(loss_dict['box'].numpy()):.4f}, "
                    f"Obj={float(loss_dict['objectness'].numpy()):.4f}, "
                    f"Cls={float(loss_dict['classification'].numpy()):.4f}"
                )

        if train_batches == 0:
            raise RuntimeError("No training batches processed.")

        avg_train_loss = train_loss_sum / train_batches
        avg_train_box = train_box_sum / train_batches
        avg_train_obj = train_obj_sum / train_batches
        avg_train_cls = train_cls_sum / train_batches

        # ====================================================
        # VALIDATION
        # ====================================================

        val_loss_sum = 0.0
        val_box_sum = 0.0
        val_obj_sum = 0.0
        val_cls_sum = 0.0

        val_batches = 0

        for images, boxes in valid_dataset:

            loss_dict = validation_step(
                images,
                boxes
            )

            val_loss_sum += float(
                loss_dict["total"].numpy()
            )

            val_box_sum += float(
                loss_dict["box"].numpy()
            )

            val_obj_sum += float(
                loss_dict["objectness"].numpy()
            )

            val_cls_sum += float(
                loss_dict["classification"].numpy()
            )

            val_batches += 1

        if val_batches == 0:
            raise RuntimeError("No validation batches processed.")

        avg_val_loss = val_loss_sum / val_batches
        avg_val_box = val_box_sum / val_batches
        avg_val_obj = val_obj_sum / val_batches
        avg_val_cls = val_cls_sum / val_batches

        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        print("\nTRAINING SUMMARY")

        print(f"Train total loss: {avg_train_loss:.6f}")
        print(f"Train box loss:   {avg_train_box:.6f}")
        print(f"Train obj loss:   {avg_train_obj:.6f}")
        print(f"Train cls loss:   {avg_train_cls:.6f}")

        print("\nVALIDATION SUMMARY")

        print(f"Valid total loss: {avg_val_loss:.6f}")
        print(f"Valid box loss:   {avg_val_box:.6f}")
        print(f"Valid obj loss:   {avg_val_obj:.6f}")
        print(f"Valid cls loss:   {avg_val_cls:.6f}")

        # ----------------------------------------------------
        # SAVE LATEST CHECKPOINT
        # ----------------------------------------------------

        model.save_weights(
            LATEST_CHECKPOINT
        )

        # ----------------------------------------------------
        # SAVE BEST CHECKPOINT
        # ----------------------------------------------------

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            model.save_weights(
                BEST_CHECKPOINT
            )

            print(
                "\nNew best validation loss:",
                f"{best_val_loss:.6f}"
            )

        print("\nLatest checkpoint saved.")

    print("\nTraining completed.")
    print("Best checkpoint:", BEST_CHECKPOINT)
    print("Latest checkpoint:", LATEST_CHECKPOINT)
