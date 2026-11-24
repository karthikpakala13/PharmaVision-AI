import tensorflow as tf
import numpy as np

ORIGINAL_PATH = (
    "/home/karthik/Downloads/tf_yolov4-512_3.5/float/"
    "yolov4-leaky.h5"
)

OUTPUT_PATH = "/home/karthik/arm2/yolov4_2class_custom.h5"

HEADS = {
    "conv2d_93": {
        "old_ids": [0, 1, 2],
        "new_ids": [0, 1, 2]
    },
    "conv2d_101": {
        "old_ids": [3, 4, 5],
        "new_ids": [3, 4, 5]
    },
    "conv2d_109": {
        "old_ids": [6, 7, 8],
        "new_ids": [6, 7, 8]
    }
}

OLD_ANCHORS = np.array([
    [12, 16], [19, 36], [40, 28],
    [36, 75], [76, 55], [72, 146],
    [142, 110], [192, 243], [459, 401]
], dtype=np.float32)

NEW_ANCHORS = np.array([
    [14, 27], [18, 32], [23, 38],
    [30, 46], [25, 58], [32, 64],
    [27, 77], [40, 53.5], [37, 78]
], dtype=np.float32)


print("Loading original YOLOv4 model...")

original = tf.keras.models.load_model(
    ORIGINAL_PATH,
    compile=False
)

# Change only the final detection-head filter counts.
def clone_layer(layer):
    config = layer.get_config()

    if layer.name in HEADS:
        config["filters"] = 21

    return layer.__class__.from_config(config)


model = tf.keras.models.clone_model(
    original,
    clone_function=clone_layer
)

print("Transferring pretrained weights...")

for layer in original.layers:

    name = layer.name

    old_weights = layer.get_weights()

    if not old_weights:
        continue

    new_layer = model.get_layer(name)

    if name not in HEADS:
        new_layer.set_weights(old_weights)
        continue

    old_kernel, old_bias = old_weights

    new_kernel, new_bias = new_layer.get_weights()

    # Copy box and objectness weights.
    # Keep newly initialized class-prediction weights.
    for slot in range(3):

        old_start = slot * 85
        new_start = slot * 7

        new_kernel[:, :, :, new_start:new_start + 5] = (
            old_kernel[:, :, :, old_start:old_start + 5]
        )

        new_bias[new_start:new_start + 5] = (
            old_bias[old_start:old_start + 5]
        )

        old_id = HEADS[name]["old_ids"][slot]
        new_id = HEADS[name]["new_ids"][slot]

        old_anchor = OLD_ANCHORS[old_id]
        new_anchor = NEW_ANCHORS[new_id]

        # Preserve initial decoded width and height.
        new_bias[new_start + 2] += np.log(
            old_anchor[0] / new_anchor[0]
        )

        new_bias[new_start + 3] += np.log(
            old_anchor[1] / new_anchor[1]
        )

    new_layer.set_weights([new_kernel, new_bias])

print("Saving corrected model...")

model.save(OUTPUT_PATH)

print("\nMODEL CREATED:", OUTPUT_PATH)

print("\nOUTPUT SHAPES")
for output in model.outputs:
    print(output.shape)

print("\nDETECTION HEADS")
for name in HEADS:
    print(name, model.get_layer(name).get_weights()[0].shape)

print("\nCustom-anchor model creation completed.")
