import os
import glob
import numpy as np
from collections import defaultdict

def bbox_iou(box1, box2):
    # box: [x_center, y_center, w, h] in YOLO format
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2
    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area + b2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def load_yolo_labels(label_file, with_conf=False):
    boxes = []
    with open(label_file, 'r') as f:
        for line in f:
            items = line.strip().split()
            if len(items) < 5:
                continue
            cls_id = int(items[0])
            box = list(map(float, items[1:5]))
            if with_conf and len(items) >= 6:
                conf = float(items[5])
                boxes.append((cls_id, box, conf))
            else:
                boxes.append((cls_id, box))
    return boxes


def compute_ap(recall, precision):
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

def calculate_map(gt_folder, pred_folder, iou_threshold=0.5, num_classes=80):
    all_gt = defaultdict(list)
    all_pred = defaultdict(list)
    gt_counter_per_class = defaultdict(int)

    gt_files = glob.glob(os.path.join(gt_folder, '*.txt'))

    for gt_file in gt_files:
        filename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_folder, filename)
        gt_boxes = load_yolo_labels(gt_file, with_conf=False)
        pred_boxes = load_yolo_labels(pred_file, with_conf=True)

        used = []
        for cls_id, box in gt_boxes:
            all_gt[cls_id].append({'file': filename, 'box': box, 'used': False})
            gt_counter_per_class[cls_id] += 1

        
        for cls_id, box, conf in pred_boxes:
            all_pred[cls_id].append({'file': filename, 'box': box, 'conf': conf})

    ap_per_class = []
    for cls in range(num_classes):
        preds = sorted(all_pred[cls], key=lambda x: x['conf'], reverse=True)
        TP = np.zeros(len(preds))
        FP = np.zeros(len(preds))
        gt_data = [x for x in all_gt[cls]]

        for i, pred in enumerate(preds):
            matched = False
            for gt in gt_data:
                if gt['file'] != pred['file'] or gt['used']:
                    continue
                iou = bbox_iou(pred['box'], gt['box'])
                if iou >= iou_threshold:
                    TP[i] = 1
                    gt['used'] = True
                    matched = True
                    break
            if not matched:
                FP[i] = 1

        cum_TP = np.cumsum(TP)
        cum_FP = np.cumsum(FP)
        total_gt = gt_counter_per_class[cls]
        if total_gt == 0:
            continue
        recall = cum_TP / total_gt
        precision = cum_TP / (cum_TP + cum_FP + 1e-6)
        ap = compute_ap(recall, precision)
        ap_per_class.append(ap)

    mAP = np.mean(ap_per_class) if ap_per_class else 0.0
    return mAP, ap_per_class

# 실행
if __name__ == "__main__":
    gt_dir = 'data/test/labels'
    pred_dir = 'runs/detect/predict/labels'
    map50, ap_per_class = calculate_map(gt_dir, pred_dir, iou_threshold=0.5)
    print(f"\n📌 mAP@0.5: {map50:.4f}")