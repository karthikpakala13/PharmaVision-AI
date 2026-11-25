# PharmaVision AI: Edge Pharmaceutical Visual Inspection & Quality Control Assistant

### ARM Bharat AI SoC Challenge 2026–27

## 1. Problem Statement & Solution Overview

Manual visual inspection on pharmaceutical packaging lines is slow, fatiguing, and prone to human error, while cloud-connected AI inspection introduces network latency, reliability risks, and strict data-privacy concerns.

**PharmaVision AI** solves this with a **100% offline, air-gapped Edge AI-SoC system** deployed on the **AMD Kria KV260**. By leveraging heterogeneous compute—offloading real-time INT8 object detection to the FPGA Deep Learning Processor Unit (DPU) while executing a 4-bit quantized Large Language Model on the Arm Cortex-A53 cores—the system delivers automated defect detection and plain-English Quality Control (QC) reports within a $\approx 5\text{ W}$ power envelope.

### Key Competition Highlights

* **Heterogeneous Edge AI Execution:** Combines hardware-accelerated **YOLOv4 (INT8 on `DPUCZDX8G`)** for visual defect detection with on-device **Meta Llama 3.2 1B (4-bit via PyTorch ExecuTorch)** on the quad-core Arm Cortex-A53 for automated QC reasoning.
* **100% Air-Gapped & Privacy-Preserving:** Runs completely offline on the KV260 with zero external API or cloud dependencies.
* **Grounded, Hallucination-Guarded QC Reporting:** Structured JSON detection metrics (counts, confidence percentages, and QC status) are passed to Llama 3.2 1B with strict prompt constraints to generate concise 3-sentence operator summaries (`NO DEFECT DETECTED` vs. `MANUAL REVIEW REQUIRED`) without fabricating unseen defect types.
* **Live Operator Console & SoC Telemetry:** A lightweight Flask web interface (`dashboard.py`) displays real-time annotated frames, downloadable inspection videos, LLM QC reports, and live onboard INA260 power (`W`), CPU, and DRAM telemetry.

---

## 2. End-to-End System Architecture

### Host Training, Quantization & Compilation Flow

```text
[Dataset Loader] ---> [Custom YOLOv4 Training (TensorFlow)] ---> [Vitis AI INT8 Quantization & Export]
                                                                                  |
[Meta Llama 3.2 1B] ---> [ExecuTorch 4-Bit Export (.pte)] -------+                v
                                                                 |   [KV260 DPU Compilation (.xmodel)]
                                                                 v                |
                                                     +---------------------------------------+
                                                     |    AMD Kria KV260 Target Deployment   |
                                                     +---------------------------------------+
```

### Target Kria KV260 Runtime Pipeline

```text
+-------------------+      +-------------------------------+      +------------------------------------+
|   USB Camera      | ---> |   YOLOv4 (512x512 INT8)       | ---> |  Detection & Confidence Aggregator |
|   (/dev/video0)   |      |   DPUCZDX8G (Vitis AI VART)   |      |  (Normal vs. Defective + JSON)     |
+-------------------+      +-------------------------------+      +------------------------------------+
                                                                                    |
                                                                                    v
+-------------------+      +-------------------------------+      +------------------------------------+
|   Flask Web UI    | <--- |   Live KV260 Telemetry        | <--- |  ExecuTorch Llama 3.2 1B (4-bit)   |
|   (Port 5000)     |      |   /proc + INA260 Power Sensor |      |  Arm Cortex-A53 (XNNPACK Backend)  |
+-------------------+      +-------------------------------+      +------------------------------------+
```

---

## 3. Repository Structure

```text
ARM-Bharat-AI-SoC-2026-27/
├── model/
│   ├── create_yolov4_custom.py         # Custom 2-class YOLOv4 network architecture
│   ├── target_assignment.py            # Anchor matching and ground-truth target encoding
│   └── yolov4_loss.py                  # Custom CIoU, objectness, and classification loss functions
├── training/
│   ├── dataset_loader.py               # Data pipeline and augmentation for Defective/Normal classes
│   ├── train_yolov4.py                 # End-to-end TensorFlow model training script
│   └── evaluate_yolov4.py              # Validation metrics (mAP, precision, recall, throughput)
├── quantization/
│   ├── convert_calibration.py          # Calibration dataset preparation for Vitis AI quantizer
│   ├── quantize_yolov4.py              # Post-training INT8 quantization script
│   └── export_vitis_model.py           # Vitis AI model export for DPU compilation
├── trained_model/
│   └── yolov4_2class_custom.h5         # Trained floating-point YOLOv4 weights
├── quantized_model/
│   ├── yolov4_2class_int8.h5           # INT8 quantized Keras/TensorFlow model
│   └── yolov4_2class_vitis_export.h5   # Exported model ready for Vitis AI compiler
├── compiled_model/
│   ├── yolov4_kv260.xmodel             # Compiled KV260 DPU executable model
│   └── meta.json                       # DPU subgraph and fingerprint metadata
├── llama/
│   ├── llama3_2.pte                    # 4-bit quantized Meta Llama 3.2 1B ExecuTorch model
│   └── tokenizer.model                 # Llama 3.2 tokenizer
├── inference_scripts/
│   ├── usb_dpu_llama.py                # Live USB capture, DPU YOLOv4 inference & ExecuTorch Llama runner
│   └── dashboard.py                    # Flask web dashboard with live telemetry & QC report viewer
├── results/
│   ├── training/
│   │   ├── test_images/                # Offline test-set detection samples
│   │   └── training_metrics/           # Training loss and validation mAP curves
│   └── deployment/                     # Live KV260 dashboard and hardware benchmark screenshots
├── docs/                               # Project documentation and presentation artifacts
└── README.md                           # Project overview and documentation
```

---

## 4. Model & Runtime Specifications

### Vision Detection Model (YOLOv4)

| Parameter | Specification |
| :--- | :--- |
| **Architecture** | Custom YOLOv4 (`model/create_yolov4_custom.py`) |
| **Target Classes (2)** | `Class 0: Defective`, `Class 1: Normal` |
| **Input Resolution** | $512 \times 512$ (aspect-ratio-preserving letterbox preprocessing) |
| **Quantization & Artifact** | INT8 quantized (`compiled_model/yolov4_kv260.xmodel`) on `DPUCZDX8G` (`DPU_0`) |
| **Inference Runtime** | Vitis AI Runtime (`VART`, `XIR`) |

### Language Reporting Model (Llama 3.2 1B)

| Parameter | Specification |
| :--- | :--- |
| **Language Model** | Meta Llama 3.2 1B Instruct |
| **Model Artifact** | ExecuTorch 4-bit quantized (`llama/llama3_2.pte`, $\approx 765\text{ MB}$) |
| **Tokenizer** | `llama/tokenizer.model` |
| **Execution Backend** | Cross-compiled ExecuTorch (`llama_main` / `XNNPACK`) on Quad-core Arm Cortex-A53 |
| **Generation Parameters** | `--seq_len 512`, `--max_new_tokens 180`, `--temperature 0.1` |

---

## 5. Evaluation & On-Board Hardware Benchmarks

### A. Offline Vision Model Evaluation (`training/evaluate_yolov4.py`)

| Evaluation Metric | Result |
| :--- | :--- |
| **mAP@0.5** | **80.50%** |
| **mAP@0.5:0.95** | **57.75%** |
| **Precision** | **76.43%** |
| **Recall** | **76.43%** |
| **Model-Only Throughput** | **10.61 FPS** |

### B. Measured Edge SoC Benchmarks (AMD Kria KV260)

> *Measured directly on the Kria KV260 board running the full offline YOLOv4 + ExecuTorch Llama 3.2 pipeline (`inference_scripts/usb_dpu_llama.py`).*

| Benchmark Metric | Measured Result | Measurement Source / Notes |
| :--- | :--- | :--- |
| **YOLOv4 DPU Throughput** | **6.3 – 6.5 FPS** ($\approx 154\text{ ms/frame}$) | `DPUCZDX8G` (`DPU_0`), $512 \times 512$ INT8 |
| **SOM Board Power** | **5.04 – 5.17 W** | Onboard INA260 power rail (`hwmon`) |
| **Compute Efficiency** | $\approx 1.22 - 1.30\text{ FPS/W}$ | DPU vision throughput normalized by SOM board power |
| **System DRAM Footprint** | **354 – 370 MB / 3930 MB** ($\approx 9.4\%$) | Resident memory on 4 GB KV260 DDR4 (`/proc/meminfo`) |
| **LLM Model Size** | $\approx 765\text{ MB}$ (`llama3_2.pte`) | 4-bit quantized ExecuTorch `XNNPACK` artifact |
| **Cloud / Network Usage** | **0 ms / 100% Offline** | Completely air-gapped on-device execution |

---

## 6. Target KV260 Deployment & Execution

### Board Directory Layout (`/home/root/`)

```text
/home/root/
├── kv260_test/
│   ├── dashboard.py                  # From inference_scripts/dashboard.py
│   ├── usb_dpu_llama.py              # From inference_scripts/usb_dpu_llama.py
│   ├── compiled_kv260/
│   │   └── yolov4_kv260.xmodel       # From compiled_model/yolov4_kv260.xmodel
│   └── results/                      # Generated inspection outputs (metrics, frame, video)
└── llama_test/
    ├── llama_main                    # Cross-compiled ExecuTorch runtime binary
    ├── llama3_2.pte                  # From llama/llama3_2.pte
    ├── tokenizer.model               # From llama/tokenizer.model
    └── lib/                          # ExecuTorch shared libraries
```

### Launch Commands (On KV260)

* **Start the Web Dashboard** (accessible at `http://<KV260_IP>:5000`):
  ```bash
  python3 /home/root/kv260_test/dashboard.py
  ```
* **Run Standalone CLI Inspection:**
  ```bash
  python3 /home/root/kv260_test/usb_dpu_llama.py --camera 0 --duration 10 --conf 0.45
  ```

---

## 7. Verification & Results Artifacts

* **`results/training/`:** Contains offline validation metrics (`training_metrics/`) and sample detection outputs (`test_images/`).
* **`results/deployment/`:** Contains live screenshots captured from the KV260 Flask dashboard demonstrating real-time `Defective` tablet/capsule bounding boxes, DPU FPS counters, INA260 board power readings, and generated Llama 3.2 1B Quality Control summaries.