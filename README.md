# Pharmaceutical Tablet Inspection Using YOLOv4

## ARM Bharat AI SoC Challenge 2026–27

### Project Overview
This project implements a two-class pharmaceutical product inspection
system using a custom YOLOv4 object detection model.

The model detects:
- Defective
- Normal

The system is developed using TensorFlow and prepared for INT8 inference
on the AMD/Xilinx Kria KV260 platform using Vitis AI.

### Project Workflow
1. Dataset preparation and annotation.
2. Custom YOLOv4 model development.
3. Model training and validation.
4. Model evaluation.
5. INT8 quantization using Vitis AI.
6. Compilation for the KV260 DPU.
7. Hardware inference using the Vitis AI runtime.

### Repository Structure
- model/: Model architecture, loss and target assignment.
- training/: Training, dataset loading and evaluation.
- quantization/: Model export, calibration and quantization.
- deployment/: KV260 inference application.
- trained_model/: Floating-point model and trained checkpoint.
- quantized_model/: Quantized and exported model files.
- compiled_model/: Compiled KV260 XMODEL and metadata.

### Dataset
The dataset contains two classes: Defective and Normal.
The dataset is not included in this repository.
Dataset access and annotation details should be provided separately
where permitted by the competition rules.

### Model Configuration
- Architecture: Custom YOLOv4
- Number of classes: 2
- Input resolution: 512 × 512
- Class 0: Defective
- Class 1: Normal

### Reported Test Results
- mAP@0.5: 80.50%
- mAP@0.5:0.95: 57.75%
- Precision: 76.43%
- Recall: 76.43%
- Model-only throughput: 10.61 FPS

These are the reported model test-evaluation results.
The throughput figure is not a KV260 hardware benchmark.

### Deployment
The compiled XMODEL is intended for execution on the Kria KV260
using the Vitis AI runtime.

The deployment environment requires the appropriate KV260 image,
DPU support, XIR and VART runtime libraries.

### Model Artifacts
The repository includes:
- Best trained checkpoint.
- Floating-point custom model.
- INT8 quantized model.
- Vitis AI exported model.
- Compiled KV260 XMODEL.

### Important
Large model files are managed using Git LFS.
The dataset and development virtual environment are not included.
