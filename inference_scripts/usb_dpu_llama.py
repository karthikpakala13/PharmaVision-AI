#!/usr/bin/env python3
"""
KV260 USB CAMERA — LETTERBOX VART YOLOv4 (512x512) + EXECUTORCH LLAMA 3.2 1B
No CSV. Outputs annotated .avi, latest_frame.jpg, latest_metrics.json, and .txt report.
"""

import os
import gc
import sys
import cv2
import json
import time
import argparse
import datetime
import subprocess
import warnings
import numpy as np

warnings.filterwarnings("ignore")

try:
    import xir
    import vart
except ImportError:
    print("xir/vart not found. Run this on the KV260 with Vitis AI runtime.")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_XMODEL  = "/home/root/kv260_test/compiled_kv260/yolov4_kv260.xmodel"
RESULTS_DIR     = "/home/root/kv260_test/results"
DEFAULT_OUTPUT  = f"{RESULTS_DIR}/usb_dpu_output.avi"
DEFAULT_REPORT  = f"{RESULTS_DIR}/usb_dpu_report.txt"
DEFAULT_FRAME   = f"{RESULTS_DIR}/latest_frame.jpg"
DEFAULT_METRICS = f"{RESULTS_DIR}/latest_metrics.json"

LLAMA_DIR       = "/home/root/llama_test"
LLAMA_BIN       = f"{LLAMA_DIR}/llama_main"
LLAMA_MODEL     = f"{LLAMA_DIR}/llama3_2.pte"
LLAMA_TOKENIZER = f"{LLAMA_DIR}/tokenizer.model"
LLAMA_PROMPT_TX = f"{LLAMA_DIR}/dpu_session_prompt.txt"

CAMERA_IDX  = 0
INPUT_SIZE  = 512
CONF_THRESH = 0.45
NMS_THRESH  = 0.2
NUM_CLASSES = 2
CLASS_NAMES = {0: "Defective", 1: "Normal"}
COLORS      = {0: (0, 0, 255), 1: (0, 255, 0)}

ANCHORS = {
    16: np.array([[116 / 512, 90 / 512], [156 / 512, 198 / 512], [373 / 512, 326 / 512]], dtype=np.float32),
    32: np.array([[30 / 512, 61 / 512], [62 / 512, 45 / 512], [59 / 512, 119 / 512]], dtype=np.float32),
    64: np.array([[10 / 512, 13 / 512], [16 / 512, 30 / 512], [33 / 512, 23 / 512]], dtype=np.float32),
}
NUM_ANCHORS = 3
FONT = cv2.FONT_HERSHEY_DUPLEX


# ══════════════════════════════════════════════════════════════════════════════
# VART & PRE/POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def get_dpu_subgraph(graph):
    root = graph.get_root_subgraph()
    child = root.toposort_child_subgraph()
    dpu = [sg for sg in child if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"]
    if not dpu:
        print("No DPU subgraph found!")
        sys.exit(1)
    return dpu[0]


def get_tensor_shape(tensor):
    return list(tensor.dims)


def letterbox(img, target_size=512, pad_color=114):
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_size - nw) // 2
    pad_y = (target_size - nh) // 2
    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def unpad_detections(detections, orig_w, orig_h, target_size, scale, pad_x, pad_y):
    fixed = []
    for d in detections:
        cx, cy, bw, bh = d["bbox"]
        px = (cx * target_size - pad_x) / scale
        py = (cy * target_size - pad_y) / scale
        pw = (bw * target_size) / scale
        ph = (bh * target_size) / scale
        fixed.append({
            "class": d["class"],
            "score": d["score"],
            "bbox": [px / orig_w, py / orig_h, pw / orig_w, ph / orig_h],
        })
    return fixed


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def decode_output(raw, grid_size, conf_thresh):
    anchors = ANCHORS[grid_size]
    h = w = grid_size
    total = h * w * NUM_ANCHORS
    pred = raw.flatten()[: total * (5 + NUM_CLASSES)].reshape(total, 5 + NUM_CLASSES)

    x_grid, y_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xy_grid = np.repeat(np.stack([x_grid, y_grid], axis=-1)[:, :, np.newaxis, :], NUM_ANCHORS, axis=2).reshape(-1, 2)
    anchor_grid = np.tile(anchors[np.newaxis, :, :], (h * w, 1, 1)).reshape(-1, 2)

    box_xy = (sigmoid(pred[:, :2]) + xy_grid) / np.array([w, h], dtype=np.float32)
    box_wh = ((sigmoid(pred[:, 2:4]) * 2.0) ** 2) * anchor_grid

    scores = sigmoid(pred[:, 4:5]) * sigmoid(pred[:, 5 : 5 + NUM_CLASSES])
    class_ids = np.argmax(scores, axis=1)
    max_scores = scores[np.arange(total), class_ids]

    mask = max_scores >= conf_thresh
    if not np.any(mask):
        return []

    boxes = np.concatenate([box_xy[mask], box_wh[mask]], axis=1)
    class_ids = class_ids[mask]
    max_scores = max_scores[mask]

    return [
        {"class": int(class_ids[i]), "score": float(max_scores[i]), "bbox": boxes[i].tolist()}
        for i in range(len(class_ids))
    ]


def iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1, iy1 = max(x1 - w1 / 2, x2 - w2 / 2), max(y1 - h1 / 2, y2 - h2 / 2)
    ix2, iy2 = min(x1 + w1 / 2, x2 + w2 / 2), min(y1 + h1 / 2, y2 + h2 / 2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (w1 * h1 + w2 * h2 - inter + 1e-6)


def nms(detections, nms_thresh):
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if iou(best["bbox"], d["bbox"]) < nms_thresh]
    return kept


def decode_all_outputs(output_data, conf_thresh, nms_thresh):
    all_dets = []
    for raw in sorted(output_data, key=lambda x: x.size):
        all_dets.extend(decode_output(raw.flatten(), raw.shape[1], conf_thresh))
    return nms(all_dets, nms_thresh)


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING & VIDEO WRITER
# ══════════════════════════════════════════════════════════════════════════════
def draw_detections(img, detections):
    h, w = img.shape[:2]
    for i, det in enumerate(detections):
        cls_id, score = det["class"], det["score"]
        cx, cy, bw, bh = det["bbox"]
        x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
        x2, y2 = min(w - 1, int((cx + bw / 2) * w)), min(h - 1, int((cy + bh / 2) * h))
        color = COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, cls_id)}: {score * 100:.1f}%"
        (lw, lh), base = cv2.getTextSize(label, FONT, 0.65, 2)
        y_lbl = max(y1, lh + 10)
        cv2.rectangle(img, (x1, y_lbl - lh - 10), (x1 + lw + 10, y_lbl + base - 10), color, -1)
        cv2.putText(img, label, (x1 + 5, y_lbl - 5), FONT, 0.65, (255, 255, 255), 2)
    return img


def draw_overlay(img, fps, detections, frame_w, class_counts):
    fh = img.shape[0]
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 32), FONT, 0.85, (0, 255, 255), 2)
    cv2.putText(img, f"Det: {len(detections)}", (10, 62), FONT, 0.7, (255, 255, 0), 2)

    if detections:
        best = max(detections, key=lambda d: d["score"])
        status_txt = f"{CLASS_NAMES.get(best['class'], best['class']).upper()} {best['score']*100:.1f}%"
        status_color = COLORS.get(best["class"], (0, 255, 255))
    else:
        status_txt, status_color = "NO DETECTION", (180, 180, 180)

    (sw, _), _ = cv2.getTextSize(status_txt, FONT, 0.85, 2)
    cv2.putText(img, status_txt, (max(10, frame_w - sw - 10), 32), FONT, 0.85, status_color, 2)

    counts_txt = f"Normal: {class_counts['Normal']} | Defective: {class_counts['Defective']}"
    cv2.putText(img, counts_txt, (10, fh - 10), FONT, 0.6, (255, 255, 255), 2)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, FONT, 0.5, 1)
    cv2.putText(img, ts, (max(10, frame_w - tw - 10), fh - 10), FONT, 0.5, (200, 200, 200), 1)
    return img


def create_writer(path, width, height, fps=15.0):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    abs_path = os.path.abspath(path)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(abs_path, cv2.CAP_FFMPEG, fourcc, fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(abs_path, fourcc, fps, (width, height))
    return writer if writer.isOpened() else None


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTORCH LLAMA 3.2 1B SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def run_llama_summary(summary_text: str) -> str:
    if not os.path.isfile(LLAMA_BIN) or not os.path.isfile(LLAMA_MODEL):
        return "[Llama skipped: llama_main or llama3_2.pte not found]"

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a pharmaceutical tablet and capsule visual inspection assistant. "
        "Summarize the inspection results below in 3 concise sentences. "
        "Keep every count, confidence percentage, and QC status exact. "
        "Do not invent physical defect types or add intro/closing filler.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + summary_text
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    with open(LLAMA_PROMPT_TX, "w", encoding="utf-8") as f:
        f.write(prompt)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LLAMA_DIR}/lib:{LLAMA_DIR}"

    cmd = [
        LLAMA_BIN,
        "--model_path", LLAMA_MODEL,
        "--tokenizer_path", LLAMA_TOKENIZER,
        "--prompt_file", LLAMA_PROMPT_TX,
        "--seq_len", "512",
        "--max_new_tokens", "180",
        "--temperature", "0.1",
    ]

    print("\nRunning ExecuTorch Llama 3.2 1B...", flush=True)
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        return f"[Llama failed with code {res.returncode}: {res.stderr.strip()}]"

    marker = "<|start_header_id|>assistant<|end_header_id|>"
    raw_out = res.stdout.split(marker)[-1] if marker in res.stdout else res.stdout
    explanation = raw_out.split("PyTorchObserver")[0].replace("<|eot_id|>", "").strip()
    return explanation or "[Llama returned empty output]"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n{'=' * 70}")
    print("  KV260 TABLET & CAPSULE INSPECTION — VART YOLOv4 + EXECUTORCH LLAMA")
    print(f"{'=' * 70}")

    graph = xir.Graph.deserialize(args.xmodel)
    runner = vart.Runner.create_runner(get_dpu_subgraph(graph), "run")

    in_shape = get_tensor_shape(runner.get_input_tensors()[0])
    input_data = [np.zeros(in_shape, dtype=np.float32)]
    output_data = [np.zeros(get_tensor_shape(t), dtype=np.float32) for t in runner.get_output_tensors()]

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Cannot open USB camera /dev/video{args.camera}")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = None if args.no_save else create_writer(args.output, frame_w, frame_h, fps=15.0)

    total_frames = 0
    frames_with_det = 0
    class_counts = {"Defective": 0, "Normal": 0}
    confidences = {"Defective": [], "Normal": []}
    best_frame = None
    best_score = -1.0
    last_frame = None
    fps_smooth = 0.0
    t_start = time.time()

    try:
        while True:
            if args.duration > 0 and (time.time() - t_start) >= args.duration:
                break

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_lb, scale, pad_x, pad_y = letterbox(img_rgb, INPUT_SIZE)
            np.copyto(input_data[0][0], img_lb.astype(np.float32) / 255.0)

            job_id = runner.execute_async(input_data, output_data)
            runner.wait(job_id)

            raw_dets = decode_all_outputs(output_data, args.conf, args.nms)
            detections = unpad_detections(raw_dets, frame_w, frame_h, INPUT_SIZE, scale, pad_x, pad_y)
            total_frames += 1

            if detections:
                frames_with_det += 1
                for d in detections:
                    cname = CLASS_NAMES.get(d["class"], f"Class_{d['class']}")
                    class_counts[cname] = class_counts.get(cname, 0) + 1
                    confidences.setdefault(cname, []).append(d["score"] * 100.0)

            out_frame = draw_detections(frame.copy(), detections)
            out_frame = draw_overlay(out_frame, fps_smooth, detections, frame_w, class_counts)
            last_frame = out_frame

            if detections:
                top_s = max(d["score"] for d in detections)
                if top_s > best_score:
                    best_score = top_s
                    best_frame = out_frame.copy()

            if writer is not None:
                writer.write(out_frame)

            elapsed = max(time.time() - t0, 1e-6)
            fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / elapsed) if fps_smooth > 0 else (1.0 / elapsed)

            det_str = ", ".join(f"{CLASS_NAMES[d['class']]} {d['score']*100:.1f}%" for d in detections) or "none"
            print(f"Frame {total_frames:4d} | FPS: {fps_smooth:5.1f} | {det_str}", flush=True)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        del runner, graph
        gc.collect()

        frame_to_save = best_frame if best_frame is not None else last_frame
        if frame_to_save is not None:
            cv2.imwrite(DEFAULT_FRAME, frame_to_save)

        duration_sec = max(time.time() - t_start, 1e-6)
        avg_fps = total_frames / duration_sec

        if frames_with_det == 0:
            qc_status = "NO OBJECTS DETECTED"
        elif class_counts["Defective"] > 0:
            qc_status = "MANUAL REVIEW REQUIRED"
        else:
            qc_status = "NO DEFECT DETECTED BY MODEL"

        norm_avg = round(sum(confidences["Normal"]) / len(confidences["Normal"]), 2) if confidences["Normal"] else 0.0
        def_avg  = round(sum(confidences["Defective"]) / len(confidences["Defective"]), 2) if confidences["Defective"] else 0.0

        summary_lines = [
            f"Duration: {duration_sec:.1f}s | Average FPS: {avg_fps:.1f}",
            f"Total frames processed: {total_frames} | Frames with detections: {frames_with_det}",
            f"Normal detections: {class_counts['Normal']} (Avg confidence: {norm_avg:.2f}%)" if class_counts["Normal"] else "Normal detections: 0 (Confidence: N/A)",
            f"Defective detections: {class_counts['Defective']} (Avg confidence: {def_avg:.2f}%)" if class_counts["Defective"] else "Defective detections: 0 (Confidence: N/A)",
            f"Overall QC Status: {qc_status}",
        ]
        summary_text = "\n".join(summary_lines)
        llama_text = "" if (args.no_llama or total_frames == 0) else run_llama_summary(summary_text)

        full_report = (
            "PHARMACEUTICAL TABLET & CAPSULE INSPECTION REPORT\n"
            "=================================================\n"
            + summary_text
            + ("\n\nLLAMA 3.2 QC SUMMARY\n====================\n" + llama_text + "\n" if llama_text else "\n")
        )
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(full_report)

        metrics = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(duration_sec, 1),
            "avg_fps": round(avg_fps, 1),
            "total_frames": total_frames,
            "frames_with_det": frames_with_det,
            "normal_count": class_counts["Normal"],
            "normal_avg_conf": norm_avg,
            "defective_count": class_counts["Defective"],
            "defective_avg_conf": def_avg,
            "qc_status": qc_status,
            "llama_summary": llama_text,
        }
        with open(DEFAULT_METRICS, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n{'=' * 70}\n{full_report.strip()}\n{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV260 Tablet & Capsule Inspection — YOLOv4 + Llama 3.2")
    parser.add_argument("--camera", type=int, default=CAMERA_IDX)
    parser.add_argument("--xmodel", type=str, default=DEFAULT_XMODEL)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    parser.add_argument("--nms", type=float, default=NMS_THRESH)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-llama", action="store_true")
    args = parser.parse_args()
    main(args)
