import tensorflow as tf
import os

MODEL_PATH = "yolov4_2class_custom.h5"
WEIGHTS_PATH = "checkpoints/best_batch2.weights.h5"
OUTPUT_PATH = "yolov4_2class_vitis_export.h5"

print("TensorFlow:", tf.__version__)

# Load architecture
print("\nLoading architecture...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Load trained weights
print("Loading trained weights...")
model.load_weights(WEIGHTS_PATH)

print("Weights loaded successfully.")

# Save complete model in legacy HDF5 format
print("\nExporting complete model...")

model.save(
    OUTPUT_PATH,
    include_optimizer=False,
    save_format="h5"
)

print("\nExport completed.")
print("Output:", OUTPUT_PATH)
print("Parameters:", model.count_params())
print("File size:", round(os.path.getsize(OUTPUT_PATH) / (1024 * 1024), 2), "MB")
