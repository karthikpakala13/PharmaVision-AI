#!/usr/bin/env python3
import os, time, argparse
import numpy as np
import tensorflow as tf
from dataset_loader import load_sample, DATASET_ROOT
from yolov4_loss import ANCHORS, SCALE_X_Y

IMAGE_SIZE = 512
CLASS_NAMES = ["defective", "normal"]
GRID_SIZES = [16, 32, 64]

def sigmoid(x):
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))

def decode_scale(raw_output, grid_size, anchors, scale_xy):
    raw = np.asarray(raw_output, dtype=np.float32).reshape(grid_size, grid_size, 3, 7)
    yy, xx = np.meshgrid(np.arange(grid_size, dtype=np.float32),
                         np.arange(grid_size, dtype=np.float32), indexing="ij")
    grid = np.stack([xx, yy], axis=-1)[:, :, None, :]
    xy = (sigmoid(raw[..., 0:2]) * scale_xy - (scale_xy - 1.0) / 2.0 + grid) / float(grid_size)
    wh = (np.exp(np.clip(raw[..., 2:4], -10.0, 10.0))
          * np.asarray(anchors, dtype=np.float32)[None, None, :, :] / float(IMAGE_SIZE))
    scores = sigmoid(raw[..., 4])[..., None] * sigmoid(raw[..., 5:7])
    return np.concatenate([xy, wh], axis=-1).reshape(-1, 4), scores.reshape(-1, 2)

def xywh_to_xyxy(boxes):
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0:2] = boxes[:, 0:2] - boxes[:, 2:4] / 2
    out[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4] / 2
    return np.clip(out, 0.0, 1.0)

def iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1, y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x2, y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    area_a = max(0, box[2]-box[0]) * max(0, box[3]-box[1])
    area_b = np.maximum(0, boxes[:, 2]-boxes[:, 0]) * np.maximum(0, boxes[:, 3]-boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)

def classwise_nms(boxes, scores, score_threshold=0.001, iou_threshold=0.5, max_detections=100):
    detections = []
    for class_id in range(2):
        ids = np.where(scores[:, class_id] >= score_threshold)[0]
        if not len(ids):
            continue
        b, s = boxes[ids], scores[ids, class_id]
        keep = tf.image.non_max_suppression(
            boxes=np.stack([b[:, 1], b[:, 0], b[:, 3], b[:, 2]], axis=1),
            scores=s, max_output_size=max_detections,
            iou_threshold=iou_threshold, score_threshold=score_threshold
        ).numpy()
        detections.extend((class_id, float(s[k]), b[k].astype(np.float32)) for k in keep)
    detections.sort(key=lambda x: x[1], reverse=True)
    return detections[:max_detections]

def average_precision(tp, fp, total_gt):
    if total_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    return float(np.mean([
        np.max(precision[recall >= r]) if np.any(recall >= r) else 0.0
        for r in np.linspace(0.0, 1.0, 101)
    ]))

def evaluate_map(predictions, ground_truth, iou_threshold):
    aps, class_aps = [], {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        gt_by_image, total_gt = {}, 0
        for image_id, gt in enumerate(ground_truth):
            boxes = gt[gt[:, 0] == class_id, 1:5]
            gt_by_image[image_id] = [boxes, np.zeros(len(boxes), dtype=bool)]
            total_gt += len(boxes)
        dets = [(score, image_id, box)
                for image_id, preds in enumerate(predictions)
                for pred_class, score, box in preds if pred_class == class_id]
        dets.sort(key=lambda x: x[0], reverse=True)
        tp, fp = [], []
        for score, image_id, box in dets:
            gt_boxes, matched = gt_by_image[image_id]
            if not len(gt_boxes):
                tp.append(0.0); fp.append(1.0); continue
            overlaps = iou_one_to_many(box, gt_boxes)
            best = int(np.argmax(overlaps))
            if overlaps[best] >= iou_threshold and not matched[best]:
                tp.append(1.0); fp.append(0.0); matched[best] = True
            else:
                tp.append(0.0); fp.append(1.0)
        ap = average_precision(np.asarray(tp), np.asarray(fp), total_gt)
        class_aps[class_name] = ap
        if total_gt:
            aps.append(ap)
    return (float(np.mean(aps)) if aps else float("nan")), class_aps

def precision_recall_at(predictions, ground_truth, confidence_threshold=0.25, iou_threshold=0.5):
    total_tp = total_fp = total_fn = 0
    for image_id, gt in enumerate(ground_truth):
        for class_id in range(2):
            gt_boxes = gt[gt[:, 0] == class_id, 1:5]
            preds = [(score, box) for c, score, box in predictions[image_id]
                     if c == class_id and score >= confidence_threshold]
            preds.sort(key=lambda x: x[0], reverse=True)
            matched = np.zeros(len(gt_boxes), dtype=bool)
            for score, box in preds:
                if not len(gt_boxes):
                    total_fp += 1; continue
                overlaps = iou_one_to_many(box, gt_boxes)
                best = int(np.argmax(overlaps))
                if overlaps[best] >= iou_threshold and not matched[best]:
                    matched[best] = True; total_tp += 1
                else:
                    total_fp += 1
            total_fn += int(np.sum(~matched))
    p = total_tp / max(total_tp + total_fp, 1)
    r = total_tp / max(total_tp + total_fn, 1)
    return p, r, total_tp, total_fp, total_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid", choices=["valid", "test"])
    parser.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "checkpoints", "best_batch2.weights.h5"))
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    args = parser.parse_args()

    for gpu in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError: pass

    model = tf.keras.models.load_model(
        os.path.join(os.path.dirname(__file__), "yolov4_2class_custom.h5"), compile=False
    )
    model.load_weights(args.checkpoint)

    split_dir = os.path.join(DATASET_ROOT, args.split)
    filenames = sorted(f for f in os.listdir(split_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not filenames:
        raise RuntimeError("No images found in " + split_dir)

    predictions, ground_truth, timings = [], [], []
    first_image, _ = load_sample(args.split, filenames[0])
    _ = model(np.expand_dims(first_image, 0), training=False)  # warm-up

    print(f"Evaluating {len(filenames)} images from {args.split}...")
    for idx, filename in enumerate(filenames):
        image, padded = load_sample(args.split, filename)
        start = time.perf_counter()
        outputs = model(np.expand_dims(image, 0), training=False)
        outputs = [x.numpy()[0] for x in outputs]  # synchronize GPU
        timings.append(time.perf_counter() - start)

        boxes_all, scores_all = [], []
        for i in range(3):
            b, s = decode_scale(outputs[i], GRID_SIZES[i], ANCHORS[i], SCALE_X_Y[i])
            boxes_all.append(b); scores_all.append(s)
        boxes = xywh_to_xyxy(np.concatenate(boxes_all))
        scores = np.concatenate(scores_all)
        predictions.append(classwise_nms(boxes, scores, args.conf, args.nms_iou))

        valid = padded[:, 0] >= 0
        gt = padded[valid].copy()
        if len(gt):
            gt[:, 1:5] = xywh_to_xyxy(gt[:, 1:5])
        ground_truth.append(gt)

        if (idx + 1) % 25 == 0 or idx + 1 == len(filenames):
            print(f"Processed {idx+1}/{len(filenames)}")

    total_gt = sum(len(x) for x in ground_truth)
    print(f"\nImages: {len(filenames)} | Ground-truth boxes: {total_gt}")
    for class_id, name in enumerate(CLASS_NAMES):
        count = sum(int(np.sum(gt[:, 0] == class_id)) for gt in ground_truth)
        print(f"GT {name}: {count}")

    maps = []
    for threshold in np.arange(0.50, 0.951, 0.05):
        m, aps = evaluate_map(predictions, ground_truth, float(threshold))
        maps.append(m)
        print(f"\nIoU {threshold:.2f}: mAP {m:.4f}")
        for name, ap in aps.items():
            print(f"  AP {name}: {ap:.4f}")

    precision, recall, tp, fp, fn = precision_recall_at(predictions, ground_truth)
    latency = float(np.mean(timings) * 1000)
    print("\n========== SUMMARY ==========")
    print(f"mAP@0.5:      {maps[0]:.4f} ({maps[0]*100:.2f}%)")
    map5095 = float(np.nanmean(maps))
    print(f"mAP@0.5:0.95: {map5095:.4f} ({map5095*100:.2f}%)")
    print(f"Precision @ confidence 0.25, IoU 0.50: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall    @ confidence 0.25, IoU 0.50: {recall:.4f} ({recall*100:.2f}%)")
    print(f"TP / FP / FN: {tp} / {fp} / {fn}")
    print(f"Mean model inference latency: {latency:.2f} ms/image")
    print(f"Model-only FPS: {1000/latency:.2f}")
    print("FPS excludes image loading, NMS and other application overhead.")

if __name__ == "__main__":
    main()
