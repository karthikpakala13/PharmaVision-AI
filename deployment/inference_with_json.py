import cv2
import numpy as np
import xir
import vart
import json

MODEL_PATH = "/home/root/kv260_test/compiled_kv260/yolov4_kv260.xmodel"
IMAGE_PATH = "/home/root/kv260_test/test_2.jpg"
OUTPUT_PATH = "/home/root/kv260_test/output.jpg"
JSON_PATH = "/home/root/kv260_test/detections.json"

IMAGE_SIZE = 512
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

CLASS_NAMES = ["Defective", "Normal"]

ANCHORS = {
    16: [[27.0, 77.0], [40.0, 53.5], [37.0, 78.0]],
    32: [[30.0, 46.0], [25.0, 58.0], [32.0, 64.0]],
    64: [[14.0, 27.0], [18.0, 32.0], [23.0, 38.0]]
}

SCALE_XY = {
    16: 1.05,
    32: 1.10,
    64: 1.20
}

FIX_POINTS = {
    "quant_conv2d_93_fix": 0,
    "quant_conv2d_101_fix": 1,
    "quant_conv2d_109_fix": 1
}


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


# Load model and create DPU runner
graph = xir.Graph.deserialize(MODEL_PATH)
root = graph.get_root_subgraph()

dpu = None

for sg in root.toposort_child_subgraph():
    if sg.has_attr("device") and sg.get_attr("device") == "DPU":
        dpu = sg
        break

if dpu is None:
    raise RuntimeError("DPU subgraph not found")

runner = vart.Runner.create_runner(dpu, "run")

# Load image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError("Unable to load test image")

original = image.copy()

# Preprocessing: BGR -> RGB, normalize to [0,1]
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
rgb = rgb.astype(np.float32) / 255.0

# Quantize input using fix point 6
input_data = np.clip(
    np.round(rgb * 64.0), -128, 127
).astype(np.int8)

input_data = np.expand_dims(input_data, axis=0)

# Allocate DPU outputs
output_tensors = runner.get_output_tensors()

output_data = [
    np.empty(t.dims, dtype=np.int8)
    for t in output_tensors
]

print("Executing YOLOv4 on KV260...")

job_id = runner.execute_async([input_data], output_data)
status = runner.wait(job_id)

if status != 0:
    raise RuntimeError("DPU inference failed: " + str(status))

print("DPU inference completed.")

# Dequantize and map outputs by tensor name
predictions = {}

for tensor, data in zip(output_tensors, output_data):
    name = tensor.name
    fix_point = FIX_POINTS[name]

    float_output = data.astype(np.float32) / (2 ** fix_point)

    grid = tensor.dims[1]
    predictions[grid] = float_output[0]

print("Output tensors dequantized.")


# Decode YOLOv4 outputs
all_boxes = {0: [], 1: []}
all_scores = {0: [], 1: []}

for grid in [16, 32, 64]:

    pred = predictions[grid].reshape(grid, grid, 3, 7)

    xy_logits = pred[..., 0:2]
    wh_logits = np.clip(pred[..., 2:4], -10, 10)
    obj_logits = pred[..., 4]
    class_logits = pred[..., 5:7]

    scale = SCALE_XY[grid]

    grid_x, grid_y = np.meshgrid(
        np.arange(grid),
        np.arange(grid)
    )

    grid_xy = np.stack([grid_x, grid_y], axis=-1)
    grid_xy = grid_xy[:, :, np.newaxis, :]

    # Decode center coordinates
    xy = sigmoid(xy_logits) * scale
    xy = xy - (scale - 1.0) / 2.0
    xy = (xy + grid_xy) / grid

    # Decode width and height
    anchors = np.array(ANCHORS[grid], dtype=np.float32)
    wh = np.exp(wh_logits) * anchors
    wh = wh / IMAGE_SIZE

    # Confidence scores
    objectness = sigmoid(obj_logits)
    class_probs = sigmoid(class_logits)

    scores = objectness[..., np.newaxis] * class_probs

    for anchor_index in range(3):
        for y in range(grid):
            for x in range(grid):

                center_x = xy[y, x, anchor_index, 0] * IMAGE_SIZE
                center_y = xy[y, x, anchor_index, 1] * IMAGE_SIZE

                box_w = wh[y, x, anchor_index, 0] * IMAGE_SIZE
                box_h = wh[y, x, anchor_index, 1] * IMAGE_SIZE

                x1 = int(center_x - box_w / 2)
                y1 = int(center_y - box_h / 2)
                x2 = int(center_x + box_w / 2)
                y2 = int(center_y + box_h / 2)

                for class_id in range(2):

                    confidence = float(
                        scores[y, x, anchor_index, class_id]
                    )

                    if confidence >= CONF_THRESHOLD:

                        box = [
                            max(0, x1),
                            max(0, y1),
                            min(IMAGE_SIZE - 1, x2),
                            min(IMAGE_SIZE - 1, y2)
                        ]

                        all_boxes[class_id].append(box)
                        all_scores[class_id].append(confidence)


# Class-wise Non-Maximum Suppression
detections = []

for class_id in range(2):

    boxes = all_boxes[class_id]
    scores = all_scores[class_id]

    if not boxes:
        continue

    xywh_boxes = [
        [b[0], b[1], b[2] - b[0], b[3] - b[1]]
        for b in boxes
    ]

    indices = cv2.dnn.NMSBoxes(
        xywh_boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    if len(indices) == 0:
        continue

    for index in np.array(indices).flatten():

        detections.append((
            class_id,
            boxes[index],
            scores[index]
        ))


# Draw detections on original image
height, width = original.shape[:2]

for class_id, box, confidence in detections:

    x1, y1, x2, y2 = box

    x1 = int(x1 * width / IMAGE_SIZE)
    y1 = int(y1 * height / IMAGE_SIZE)
    x2 = int(x2 * width / IMAGE_SIZE)
    y2 = int(y2 * height / IMAGE_SIZE)

    label = "{}: {:.2f}".format(
        CLASS_NAMES[class_id], confidence
    )

    cv2.rectangle(
        original,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        original,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

cv2.imwrite(OUTPUT_PATH, original)

# Export detections in ORIGINAL IMAGE coordinates.
json_detections = []
for class_id, box, confidence in detections:
    x1, y1, x2, y2 = box
    original_box = {
        "x1": int(round(x1 * width / IMAGE_SIZE)),
        "y1": int(round(y1 * height / IMAGE_SIZE)),
        "x2": int(round(x2 * width / IMAGE_SIZE)),
        "y2": int(round(y2 * height / IMAGE_SIZE))
    }

    # Keep coordinates inside the original image.
    original_box["x1"] = max(0, min(width - 1, original_box["x1"]))
    original_box["y1"] = max(0, min(height - 1, original_box["y1"]))
    original_box["x2"] = max(0, min(width - 1, original_box["x2"]))
    original_box["y2"] = max(0, min(height - 1, original_box["y2"]))

    json_detections.append({
        "class_id": int(class_id),
        "class_name": CLASS_NAMES[class_id],
        "confidence": round(float(confidence), 6),
        "bounding_box": original_box
    })

summary = {
    "Defective": sum(1 for d in json_detections if d["class_name"] == "Defective"),
    "Normal": sum(1 for d in json_detections if d["class_name"] == "Normal")
}

result = {
    "image": IMAGE_PATH,
    "model": MODEL_PATH,
    "image_width": int(width),
    "image_height": int(height),
    "confidence_threshold": CONF_THRESHOLD,
    "nms_threshold": NMS_THRESHOLD,
    "total_detections": len(json_detections),
    "class_counts": summary,
    "detections": json_detections
}

with open(JSON_PATH, "w") as json_file:
    json.dump(result, json_file, indent=4)

print("\\nPOST-PROCESSING COMPLETED")
print("Total detections:", len(detections))

for class_id in range(2):
    count = sum(1 for d in detections if d[0] == class_id)
    print(CLASS_NAMES[class_id], ":", count)

print("\\nINDIVIDUAL DETECTIONS")
for index, detection in enumerate(json_detections, start=1):
    print(
        "{}. {} | confidence={} | bbox={}".format(
            index,
            detection["class_name"],
            detection["confidence"],
            detection["bounding_box"]
        )
    )

print("\\nSaved image:", OUTPUT_PATH)
print("Saved detection JSON:", JSON_PATH)
