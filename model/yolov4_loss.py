import tensorflow as tf
import numpy as np

IMAGE_SIZE = 512
NUM_CLASSES = 2
EPSILON = 1e-7

# Anchor order: 16x16, 32x32, 64x64
ANCHORS = [
    [[27, 77], [40, 53.5], [37, 78]],
    [[30, 46], [25, 58], [32, 64]],
    [[14, 27], [18, 32], [23, 38]]
]

SCALE_X_Y = [1.05, 1.1, 1.2]

IOU_NORMALIZER = 0.07
CLS_NORMALIZER = 1.0
OBJ_NORMALIZER = 1.0
IGNORE_THRESHOLD = 0.7


# --------------------------------------------------
# BOX CONVERSION
# --------------------------------------------------

def xywh_to_xyxy(boxes):

    center = boxes[..., 0:2]
    half_size = boxes[..., 2:4] / 2.0

    return tf.concat(
        [center - half_size, center + half_size],
        axis=-1
    )


# --------------------------------------------------
# PAIRWISE IOU
# --------------------------------------------------

def pairwise_iou(pred_boxes, gt_boxes):

    pred_xyxy = xywh_to_xyxy(pred_boxes)
    gt_xyxy = xywh_to_xyxy(gt_boxes)

    intersection_min = tf.maximum(
        pred_xyxy[:, :, None, :2],
        gt_xyxy[:, None, :, :2]
    )

    intersection_max = tf.minimum(
        pred_xyxy[:, :, None, 2:],
        gt_xyxy[:, None, :, 2:]
    )

    intersection_wh = tf.maximum(
        intersection_max - intersection_min,
        0.0
    )

    intersection_area = (
        intersection_wh[..., 0]
        * intersection_wh[..., 1]
    )

    pred_area = (
        pred_boxes[..., 2] * pred_boxes[..., 3]
    )[:, :, None]

    gt_area = (
        gt_boxes[..., 2] * gt_boxes[..., 3]
    )[:, None, :]

    union = pred_area + gt_area - intersection_area

    return intersection_area / tf.maximum(
        union,
        EPSILON
    )


# --------------------------------------------------
# ALIGNED IOU AND CIOU
# --------------------------------------------------

def aligned_iou_and_ciou(pred_boxes, true_boxes):

    pred_xyxy = xywh_to_xyxy(pred_boxes)
    true_xyxy = xywh_to_xyxy(true_boxes)

    intersection_min = tf.maximum(
        pred_xyxy[..., :2],
        true_xyxy[..., :2]
    )

    intersection_max = tf.minimum(
        pred_xyxy[..., 2:],
        true_xyxy[..., 2:]
    )

    intersection_wh = tf.maximum(
        intersection_max - intersection_min,
        0.0
    )

    intersection = (
        intersection_wh[..., 0]
        * intersection_wh[..., 1]
    )

    pred_area = (
        pred_boxes[..., 2] * pred_boxes[..., 3]
    )

    true_area = (
        true_boxes[..., 2] * true_boxes[..., 3]
    )

    union = pred_area + true_area - intersection

    iou = intersection / tf.maximum(
        union,
        EPSILON
    )

    center_distance = tf.reduce_sum(
        tf.square(
            pred_boxes[..., :2] - true_boxes[..., :2]
        ),
        axis=-1
    )

    enclosing_min = tf.minimum(
        pred_xyxy[..., :2],
        true_xyxy[..., :2]
    )

    enclosing_max = tf.maximum(
        pred_xyxy[..., 2:],
        true_xyxy[..., 2:]
    )

    enclosing_wh = enclosing_max - enclosing_min

    enclosing_diagonal = tf.reduce_sum(
        tf.square(enclosing_wh),
        axis=-1
    )

    pred_angle = tf.atan(
        pred_boxes[..., 2] /
        tf.maximum(pred_boxes[..., 3], EPSILON)
    )

    true_angle = tf.atan(
        true_boxes[..., 2] /
        tf.maximum(true_boxes[..., 3], EPSILON)
    )

    v = (
        4.0 / (np.pi ** 2)
    ) * tf.square(true_angle - pred_angle)

    alpha = tf.stop_gradient(
        v / tf.maximum(
            1.0 - iou + v,
            EPSILON
        )
    )

    ciou = (
        iou
        - center_distance /
        tf.maximum(enclosing_diagonal, EPSILON)
        - alpha * v
    )

    return iou, ciou


# --------------------------------------------------
# DECODE PREDICTIONS
# --------------------------------------------------

def decode_predictions(
    raw_output,
    grid_size,
    anchors,
    scale_x_y
):

    batch_size = tf.shape(raw_output)[0]

    raw = tf.reshape(
        raw_output,
        [batch_size, grid_size, grid_size, 3, 7]
    )

    xy_logits = raw[..., 0:2]
    wh_logits = raw[..., 2:4]

    objectness_logits = raw[..., 4]
    class_logits = raw[..., 5:7]

    grid_x, grid_y = tf.meshgrid(
        tf.range(grid_size, dtype=tf.float32),
        tf.range(grid_size, dtype=tf.float32)
    )

    grid = tf.stack(
        [grid_x, grid_y],
        axis=-1
    )

    grid = grid[None, :, :, None, :]

    xy = (
        tf.sigmoid(xy_logits) * scale_x_y
        - (scale_x_y - 1.0) / 2.0
        + grid
    ) / float(grid_size)

    anchor_tensor = tf.constant(
        anchors,
        dtype=tf.float32
    )

    wh_logits = tf.clip_by_value(
        wh_logits,
        -10.0,
        10.0
    )

    wh = (
        tf.exp(wh_logits)
        * anchor_tensor[None, None, None, :, :]
        / float(IMAGE_SIZE)
    )

    boxes = tf.concat(
        [xy, wh],
        axis=-1
    )

    return boxes, objectness_logits, class_logits


# --------------------------------------------------
# DECODE TARGETS
# --------------------------------------------------

def decode_targets(target, grid_size, anchors):

    batch_size = tf.shape(target)[0]

    target = tf.reshape(
        target,
        [batch_size, grid_size, grid_size, 3, 7]
    )

    tx_ty = target[..., 0:2]
    tw_th = target[..., 2:4]

    grid_x, grid_y = tf.meshgrid(
        tf.range(grid_size, dtype=tf.float32),
        tf.range(grid_size, dtype=tf.float32)
    )

    grid = tf.stack(
        [grid_x, grid_y],
        axis=-1
    )

    grid = grid[None, :, :, None, :]

    xy = (
        grid + tx_ty
    ) / float(grid_size)

    anchor_tensor = tf.constant(
        anchors,
        dtype=tf.float32
    )

    wh = (
        tf.exp(
            tf.clip_by_value(
                tw_th,
                -10.0,
                10.0
            )
        )
        * anchor_tensor[None, None, None, :, :]
        / float(IMAGE_SIZE)
    )

    return tf.concat(
        [xy, wh],
        axis=-1
    )


# --------------------------------------------------
# SINGLE SCALE LOSS WITH DIAGNOSTICS
# --------------------------------------------------

def single_scale_loss(
    raw_output,
    target,
    gt_boxes,
    grid_size,
    anchors,
    scale_x_y
):

    batch_size = tf.shape(raw_output)[0]

    decoded_boxes, obj_logits, cls_logits = (
        decode_predictions(
            raw_output,
            grid_size,
            anchors,
            scale_x_y
        )
    )

    target_boxes = decode_targets(
        target,
        grid_size,
        anchors
    )

    positive_mask = target[..., 4] > 0.5

    pred_boxes = tf.reshape(
        decoded_boxes,
        [batch_size, -1, 4]
    )

    true_boxes = tf.reshape(
        target_boxes,
        [batch_size, -1, 4]
    )

    positive_flat = tf.reshape(
        positive_mask,
        [batch_size, -1]
    )

    obj_logits = tf.reshape(
        obj_logits,
        [batch_size, -1]
    )

    cls_logits = tf.reshape(
        cls_logits,
        [batch_size, -1, NUM_CLASSES]
    )

    target_classes = tf.reshape(
        target[..., 5:7],
        [batch_size, -1, NUM_CLASSES]
    )

    # ----------------------------------------------
    # BOX LOSS
    # ----------------------------------------------

    iou, ciou = aligned_iou_and_ciou(
        pred_boxes,
        true_boxes
    )

    box_scale = (
        2.0 - true_boxes[..., 2] * true_boxes[..., 3]
    )

    box_loss_values = (
        (1.0 - ciou)
        * box_scale
        * tf.cast(positive_flat, tf.float32)
    )

    box_loss = (
        IOU_NORMALIZER
        * tf.reduce_sum(box_loss_values)
        / tf.cast(batch_size, tf.float32)
    )

    # ----------------------------------------------
    # GROUND-TRUTH BOXES
    # ----------------------------------------------

    gt_valid = gt_boxes[..., 0] >= 0

    gt_xywh = gt_boxes[..., 1:5]

    gt_xywh = tf.where(
        gt_valid[..., None],
        gt_xywh,
        tf.zeros_like(gt_xywh)
    )

    # ----------------------------------------------
    # IGNORE MASK
    # ----------------------------------------------

    all_ious = pairwise_iou(
        pred_boxes,
        gt_xywh
    )

    all_ious = tf.where(
        gt_valid[:, None, :],
        all_ious,
        tf.zeros_like(all_ious)
    )

    best_iou = tf.reduce_max(
        all_ious,
        axis=-1
    )

    negative_mask = (
        best_iou < IGNORE_THRESHOLD
    )

    # Exclude positive predictions from negative count.
    selected_negative_mask = tf.logical_and(
        tf.logical_not(positive_flat),
        negative_mask
    )

    objectness_mask = tf.logical_or(
        positive_flat,
        selected_negative_mask
    )

    # ----------------------------------------------
    # OBJECTNESS TARGET
    # ----------------------------------------------

    objectness_target = tf.stop_gradient(
        tf.clip_by_value(
            iou,
            0.0,
            1.0
        )
    )

    objectness_target = tf.where(
        positive_flat,
        objectness_target,
        tf.zeros_like(objectness_target)
    )

    # ----------------------------------------------
    # OBJECTNESS BCE
    # ----------------------------------------------

    objectness_bce = (
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=objectness_target,
            logits=obj_logits
        )
    )

    # Separate positive and negative contributions.
    positive_objectness_loss = (
        OBJ_NORMALIZER
        * tf.reduce_sum(
            objectness_bce
            * tf.cast(positive_flat, tf.float32)
        )
        / tf.cast(batch_size, tf.float32)
    )

    negative_objectness_loss = (
        OBJ_NORMALIZER
        * tf.reduce_sum(
            objectness_bce
            * tf.cast(
                selected_negative_mask,
                tf.float32
            )
        )
        / tf.cast(batch_size, tf.float32)
    )

    # Preserve the original objectness loss calculation.
    objectness_bce *= tf.cast(
        objectness_mask,
        tf.float32
    )

    objectness_loss = (
        OBJ_NORMALIZER
        * tf.reduce_sum(objectness_bce)
        / tf.cast(batch_size, tf.float32)
    )

    # ----------------------------------------------
    # CLASSIFICATION LOSS
    # ----------------------------------------------

    classification_bce = (
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target_classes,
            logits=cls_logits
        )
    )

    classification_bce = tf.reduce_sum(
        classification_bce,
        axis=-1
    )

    positive_flat_float = tf.cast(
        positive_flat,
        tf.float32
    )

    classification_bce *= positive_flat_float

    classification_loss = (
        CLS_NORMALIZER
        * tf.reduce_sum(classification_bce)
        / tf.cast(batch_size, tf.float32)
    )

    # ----------------------------------------------
    # TOTAL LOSS
    # ----------------------------------------------

    total_loss = (
        box_loss
        + objectness_loss
        + classification_loss
    )

    # ----------------------------------------------
    # DIAGNOSTICS
    # ----------------------------------------------

    positive_count = tf.reduce_sum(
        tf.cast(positive_flat, tf.float32)
    )

    negative_count = tf.reduce_sum(
        tf.cast(
            selected_negative_mask,
            tf.float32
        )
    )

    return {

        "total": total_loss,

        "box": box_loss,

        "objectness": objectness_loss,

        "classification": classification_loss,

        "positive_objectness": positive_objectness_loss,

        "negative_objectness": negative_objectness_loss,

        "positive_count": positive_count,

        "negative_count": negative_count
    }


# --------------------------------------------------
# THREE-SCALE YOLOv4 LOSS
# --------------------------------------------------

def yolov4_loss(predictions, targets, gt_boxes):

    grid_sizes = [16, 32, 64]

    scale_losses = []

    for i in range(3):

        loss = single_scale_loss(

            raw_output=predictions[i],

            target=targets[i],

            gt_boxes=gt_boxes,

            grid_size=grid_sizes[i],

            anchors=ANCHORS[i],

            scale_x_y=SCALE_X_Y[i]
        )

        scale_losses.append(loss)

    keys = [
        "total",
        "box",
        "objectness",
        "classification",
        "positive_objectness",
        "negative_objectness",
        "positive_count",
        "negative_count"
    ]

    return {
        key: tf.add_n([
            loss[key] for loss in scale_losses
        ])
        for key in keys
    }


# --------------------------------------------------
# BASIC TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("\nTESTING YOLOv4 LOSS WITH DIAGNOSTICS")
    print("=" * 55)

    predictions = [

        tf.random.normal(
            [1, 16, 16, 21]
        ),

        tf.random.normal(
            [1, 32, 32, 21]
        ),

        tf.random.normal(
            [1, 64, 64, 21]
        )
    ]

    targets = [

        tf.zeros(
            [1, 16, 16, 3, 7]
        ),

        tf.zeros(
            [1, 32, 32, 3, 7]
        ),

        tf.zeros(
            [1, 64, 64, 3, 7]
        )
    ]

    # One positive target per scale.
    targets = [

        tf.tensor_scatter_nd_update(
            targets[i],

            indices=[
                [0, 0, 0, 0]
            ],

            updates=[[
                0.5,
                0.5,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0
            ]]
        )

        for i in range(3)
    ]

    gt_boxes = tf.constant(

        [[

            [
                0.0,
                0.03125,
                0.03125,
                14.0 / 512,
                27.0 / 512
            ],

            [
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0
            ]

        ]],

        dtype=tf.float32
    )

    result = yolov4_loss(
        predictions,
        targets,
        gt_boxes
    )

    for name, value in result.items():

        print(
            f"{name}: {float(value.numpy()):.6f}"
        )

        assert np.isfinite(
            value.numpy()
        ), f"Invalid {name} loss"

    # Verify that positive and negative losses
    # reconstruct the original objectness loss.
    reconstructed_objectness = (
        result["positive_objectness"]
        + result["negative_objectness"]
    )

    np.testing.assert_allclose(
        result["objectness"].numpy(),
        reconstructed_objectness.numpy(),
        rtol=1e-5,
        atol=1e-5
    )

    print("\nOBJECTNESS DIAGNOSTICS TEST PASSED.")
