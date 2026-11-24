import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.quantization.keras import vitis_quantize

MODEL_PATH = "/workspace/yolov4_2class_vitis_export.h5"
CALIB_DIR = "/workspace/calibration_images_512"
ARCH_PATH = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"
OUTPUT_PATH = "/workspace/yolov4_2class_int8.h5"

IMAGE_SIZE = 512

print("TensorFlow version:", tf.__version__)

# 1. Load FP32 model
print("\nLoading floating-point model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Model loaded.")
print("Input:", model.input_shape)
print("Outputs:", [x.shape for x in model.outputs])

# 2. Collect calibration images
filenames = sorted(
    f for f in os.listdir(CALIB_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

if len(filenames) != 100:
    raise RuntimeError(
        f"Expected 100 calibration images, found {len(filenames)}"
    )

def calibration_generator():
    for filename in filenames:
        path = os.path.join(CALIB_DIR, filename)
        image = cv2.imread(path)

        if image is None:
            raise ValueError("Cannot read: " + path)

        if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Incorrect image size: " + path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        yield image

calib_dataset = tf.data.Dataset.from_generator(
    calibration_generator,
    output_signature=tf.TensorSpec(
        shape=(512, 512, 3),
        dtype=tf.float32
    )
).batch(1)

print("\nCalibration images:", len(filenames))

# 3. Verify calibration input
sample_batch = next(iter(calib_dataset))
print("Calibration batch shape:", sample_batch.shape)
print("Pixel range:",
      float(tf.reduce_min(sample_batch)),
      "to",
      float(tf.reduce_max(sample_batch)))

# 4. Initialize KV260-targeted quantizer
print("\nInitializing Vitis AI quantizer...")

quantizer = vitis_quantize.VitisQuantizer(
    model,
    quantize_strategy="pof2s",
    target=ARCH_PATH,
    target_type="json"
)

# 5. Perform INT8 post-training quantization
print("\nStarting INT8 quantization...")
quantized_model = quantizer.quantize_model(
    calib_dataset=calib_dataset,
    verbose=1
)

# 6. Save quantized model
print("\nSaving INT8 model...")
quantized_model.save(
    OUTPUT_PATH,
    include_optimizer=False,
    save_format="h5"
)

print("\nINT8 QUANTIZATION COMPLETED.")
print("Output:", OUTPUT_PATH)
print("File size:", round(os.path.getsize(OUTPUT_PATH) / (1024 * 1024), 2), "MB")
