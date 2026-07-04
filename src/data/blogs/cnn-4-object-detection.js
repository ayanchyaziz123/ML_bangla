export const cnn_4_object_detection = {
  title: "Object Detection: R-CNN থেকে YOLO — ছবিতে বস্তু খোঁজার বিজ্ঞান",
  description: "Image classification, detection ও segmentation-এর পার্থক্য, R-CNN পরিবার (R-CNN, Fast R-CNN, Faster R-CNN), YOLO-র single-pass architecture, anchor boxes, NMS এবং mAP metric — সম্পূর্ণ কোড সহ।",
  date: "২৩ মে, ২০২৬",
  category: "কনভোলিউশনাল নিউরাল নেটওয়ার্ক",
  readTime: 13,
  slug: "cnn-object-detection",
  content: `
    <h3>১. Classification, Detection ও Segmentation: পার্থক্য কোথায়?</h3>
    <p>Computer Vision-এর তিনটি প্রধান task:</p>
    <table>
      <thead>
        <tr><th>Task</th><th>Output</th><th>উদাহরণ</th><th>Complexity</th></tr>
      </thead>
      <tbody>
        <tr><td>Image Classification</td><td>Single class label</td><td>"এটি একটি বিড়াল"</td><td>Low</td></tr>
        <tr><td>Object Detection</td><td>Class + Bounding Box(es)</td><td>"বিড়াল at [x1,y1,x2,y2]"</td><td>Medium</td></tr>
        <tr><td>Semantic Segmentation</td><td>Per-pixel class label</td><td>প্রতিটি pixel = বিড়াল/পটভূমি</td><td>High</td></tr>
        <tr><td>Instance Segmentation</td><td>Per-object mask + class</td><td>বিড়াল-১ mask, বিড়াল-২ mask আলাদা</td><td>Very High</td></tr>
        <tr><td>Panoptic Segmentation</td><td>Semantic + Instance combined</td><td>সব object + background label</td><td>Highest</td></tr>
      </tbody>
    </table>
    <p><strong>Bounding Box (bbox):</strong> একটি object-কে ঘেরা আয়তক্ষেত্র। সাধারণত [x_center, y_center, width, height] বা [x_min, y_min, x_max, y_max] format-এ।</p>

    <h3>২. Sliding Window: সরল কিন্তু ধীর পদ্ধতি</h3>
    <p>প্রথম approach: image-এর বিভিন্ন scale ও position-এ ছোট window slide করো, প্রতিটিতে classifier চালাও। সমস্যা:</p>
    <ul>
      <li><strong>অত্যন্ত ধীর:</strong> 1000×1000 image-এ হাজার হাজার windows</li>
      <li><strong>Aspect ratio সমস্যা:</strong> লম্বা বা চওড়া object miss হয়</li>
      <li><strong>Scale variation:</strong> দূরের ও কাছের object-এর জন্য multiple scales</li>
    </ul>
    <pre><code>import numpy as np

def sliding_window_positions(image_h, image_w, window_h, window_w, stride=8):
    """Sliding window position generator"""
    positions = []
    for y in range(0, image_h - window_h + 1, stride):
        for x in range(0, image_w - window_w + 1, stride):
            positions.append((x, y, x + window_w, y + window_h))
    return positions

# উদাহরণ: 640x480 image, 128x128 window, stride=32
positions = sliding_window_positions(480, 640, 128, 128, stride=32)
print(f"Total windows to evaluate: {len(positions)}")
# Computational cost কত বেশি তা বোঝা যাচ্ছে!

# IoU (Intersection over Union) calculation
def compute_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    # Intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_width  = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    # Union
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)

# Test IoU
pred_box  = [50, 50, 150, 150]
truth_box = [60, 60, 160, 160]
iou = compute_iou(pred_box, truth_box)
print(f"IoU: {iou:.4f}")  # ~0.68
</code></pre>

    <h3>৩. R-CNN পরিবার: Region Proposal + CNN</h3>
    <p><strong>R-CNN (2014):</strong> Selective Search দিয়ে ~2000 region proposal → প্রতিটি warp করে CNN-এ দাও → SVM classify → Bbox regression।</p>
    <p>সমস্যা: 2000 forward pass → test-এ 47 seconds per image!</p>
    <p><strong>Fast R-CNN (2015):</strong> পুরো image একবার CNN-এ দাও → feature map পাও → RoI Pooling দিয়ে প্রতিটি region-এর feature extract → একসাথে classify ও bbox regression।</p>
    <p>গতি: ~2 seconds — bottleneck এখন selective search।</p>
    <p><strong>Faster R-CNN (2015):</strong> Region Proposal Network (RPN) যোগ করো — CNN features থেকেই proposal generate হয়। End-to-end trainable।</p>
    <table>
      <thead>
        <tr><th>Method</th><th>Test Speed</th><th>mAP (VOC07)</th><th>Bottleneck</th></tr>
      </thead>
      <tbody>
        <tr><td>R-CNN</td><td>47 sec/img</td><td>58.5%</td><td>2000 CNN forward passes</td></tr>
        <tr><td>Fast R-CNN</td><td>2.3 sec/img</td><td>70.0%</td><td>Selective search (CPU)</td></tr>
        <tr><td>Faster R-CNN</td><td>0.2 sec/img</td><td>73.2%</td><td>RPN overhead</td></tr>
        <tr><td>YOLOv3</td><td>0.022 sec/img</td><td>57.9% (COCO)</td><td>Accuracy tradeoff</td></tr>
      </tbody>
    </table>
    <pre><code>import torch
import torch.nn.functional as F

# RoI Pooling concept (simplified)
def roi_pooling(feature_map, rois, output_size=(7, 7)):
    """
    feature_map: (C, H, W) feature tensor
    rois: list of [y1, x1, y2, x2] normalized coordinates
    output_size: fixed output size for each RoI
    """
    pooled_rois = []
    C, fH, fW = feature_map.shape

    for roi in rois:
        y1, x1, y2, x2 = roi
        # Map to feature map coordinates
        fy1 = int(y1 * fH)
        fx1 = int(x1 * fW)
        fy2 = int(y2 * fH)
        fx2 = int(x2 * fW)

        roi_feature = feature_map[:, fy1:fy2, fx1:fx2]

        if roi_feature.shape[1] == 0 or roi_feature.shape[2] == 0:
            pooled_rois.append(torch.zeros(C, *output_size))
            continue

        # Adaptive max pooling to fixed output_size
        pooled = F.adaptive_max_pool2d(roi_feature.unsqueeze(0), output_size)
        pooled_rois.append(pooled.squeeze(0))

    return torch.stack(pooled_rois)

# উদাহরণ
feature_map = torch.randn(512, 14, 14)
rois = [
    [0.1, 0.1, 0.5, 0.5],  # RoI 1
    [0.4, 0.4, 0.9, 0.9],  # RoI 2
]

pooled = roi_pooling(feature_map, rois)
print(f"RoI Pooling output: {tuple(pooled.shape)}")  # (2, 512, 7, 7)
</code></pre>

    <h3>৪. YOLO: You Only Look Once</h3>
    <p>YOLO (Redmon et al., 2016) সম্পূর্ণ আলাদা approach: একটি single CNN forward pass-এ সব detection করো।</p>
    <p><strong>Grid System:</strong> Image-কে S×S grid-এ ভাগ করো (যেমন 13×13)। প্রতিটি grid cell B টি bounding box predict করে।</p>
    <p>প্রতিটি grid cell output: B × (5 + C) values</p>
    <ul>
      <li>5 = [x_center, y_center, width, height, objectness_score]</li>
      <li>C = number of classes</li>
    </ul>
    <p><strong>YOLO Loss Function:</strong></p>
    <p>L = λ_coord × Σ box regression loss + λ_noobj × Σ no-object loss + Σ class loss</p>
    <p><strong>Anchor Boxes:</strong> Pre-defined aspect ratios ও sizes। প্রতিটি cell প্রতিটি anchor-এর জন্য offset predict করে।</p>
    <pre><code>import numpy as np

# YOLO output interpretation
def decode_yolo_output(raw_output, anchors, num_classes, input_size=416):
    """
    raw_output: (batch, grid_h, grid_w, num_anchors, 5+num_classes)
    anchors: list of (w, h) anchor box sizes
    """
    batch_size, grid_h, grid_w, num_anchors, _ = raw_output.shape

    # Decode predictions
    # Box center: sigmoid activation → [0, 1] relative to grid cell
    box_xy = 1 / (1 + np.exp(-raw_output[..., :2]))  # sigmoid

    # Box size: exponential → scale anchors
    box_wh = np.exp(raw_output[..., 2:4])

    # Objectness: sigmoid
    objectness = 1 / (1 + np.exp(-raw_output[..., 4:5]))

    # Class probabilities: softmax
    class_probs_raw = raw_output[..., 5:]
    exp_scores = np.exp(class_probs_raw - np.max(class_probs_raw, axis=-1, keepdims=True))
    class_probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Grid offset যোগ করা
    grid_x = np.arange(grid_w).reshape(1, 1, grid_w, 1)
    grid_y = np.arange(grid_h).reshape(1, grid_h, 1, 1)

    box_x = (box_xy[..., 0:1] + grid_x) / grid_w
    box_y = (box_xy[..., 1:2] + grid_y) / grid_h

    # Anchor scaling
    anchors_arr = np.array(anchors).reshape(1, 1, 1, num_anchors, 2)
    box_w = box_wh[..., 0:1] * anchors_arr[..., 0:1] / input_size
    box_h = box_wh[..., 1:2] * anchors_arr[..., 1:2] / input_size

    confidence = objectness * class_probs
    return box_x, box_y, box_w, box_h, confidence

# Anchors for YOLOv3 (COCO) — 3 scales × 3 anchors
anchors_small  = [(10,13), (16,30), (33,23)]
anchors_medium = [(30,61), (62,45), (59,119)]
anchors_large  = [(116,90), (156,198), (373,326)]

print("YOLO uses multi-scale detection:")
print(f"  Small objects: 52x52 grid, anchors {anchors_small}")
print(f"  Medium objects: 26x26 grid, anchors {anchors_medium}")
print(f"  Large objects: 13x13 grid, anchors {anchors_large}")
</code></pre>

    <h3>৫. Non-Maximum Suppression (NMS)</h3>
    <p>YOLO একই object-এর জন্য multiple detection করে। NMS duplicate boxes সরিয়ে সেরা box রাখে।</p>
    <p><strong>NMS Algorithm:</strong></p>
    <ol>
      <li>Confidence score অনুযায়ী sort করো</li>
      <li>সবচেয়ে confident box রাখো</li>
      <li>বাকি boxes-এর সাথে IoU calculate করো</li>
      <li>IoU threshold-এর বেশি হলে সেই box সরাও (duplicate)</li>
      <li>Repeat</li>
    </ol>
    <pre><code>import numpy as np

def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    boxes: (N, 4) array [x1, y1, x2, y2]
    scores: (N,) confidence scores
    returns: indices of kept boxes
    """
    if len(boxes) == 0:
        return []

    # Score অনুযায়ী sort (descending)
    sorted_indices = np.argsort(scores)[::-1]
    keep = []

    while len(sorted_indices) > 0:
        # সবচেয়ে confident box রাখো
        current = sorted_indices[0]
        keep.append(current)

        if len(sorted_indices) == 1:
            break

        # বাকি সব box-এর সাথে IoU compute করো
        remaining = sorted_indices[1:]
        ious = np.array([
            compute_iou(boxes[current], boxes[i])
            for i in remaining
        ])

        # IoU threshold-এর নিচের boxes রাখো
        below_threshold = remaining[ious &lt; iou_threshold]
        sorted_indices = below_threshold

    return keep

def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    intersection = max(0, xi2-xi1) * max(0, yi2-yi1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)

# Test NMS
boxes = np.array([
    [100, 100, 200, 200],
    [110, 110, 210, 210],  # Highly overlapping with box 0
    [300, 300, 400, 400],  # Different location
    [105, 105, 205, 205],  # Highly overlapping with box 0
])
scores = np.array([0.9, 0.75, 0.8, 0.6])

kept_indices = nms(boxes, scores, iou_threshold=0.5)
print(f"Original boxes: {len(boxes)}")
print(f"After NMS: {len(kept_indices)} boxes, indices: {kept_indices}")
# Box 0 (0.9) ও Box 2 (0.8) রাখবে, overlapping গুলো সরাবে
</code></pre>

    <h3>৬. YOLOv8 দিয়ে Inference (Ultralytics)</h3>
    <pre><code># pip install ultralytics আগে install করতে হবে
from ultralytics import YOLO
import cv2
import numpy as np

# Pre-trained YOLOv8 model load (n=nano, s=small, m=medium, l=large, x=extra-large)
model = YOLO('yolov8n.pt')  # Smallest, fastest model

# Webcam বা image-এ inference
results = model('https://ultralytics.com/images/bus.jpg')

# Results parse করা
for result in results:
    boxes      = result.boxes       # Bounding boxes
    masks      = result.masks       # Segmentation masks (যদি থাকে)
    keypoints  = result.keypoints   # Pose keypoints (যদি থাকে)
    probs      = result.probs       # Classification probabilities

    print(f"Detected {len(boxes)} objects")
    for box in boxes:
        cls_id     = int(box.cls[0])
        conf       = float(box.conf[0])
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        class_name = result.names[cls_id]
        print(f"  {class_name}: {conf:.2f} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

# Custom training
# model = YOLO('yolov8n.yaml')  # New model from scratch
# model.train(data='coco128.yaml', epochs=100, imgsz=640, batch=16)

# Export to ONNX for deployment
# model.export(format='onnx')

# Real-time video detection
# model.predict(source=0, show=True, stream=True)  # source=0 = webcam
</code></pre>

    <h3>৭. Detection Metrics: IoU ও mAP</h3>
    <p><strong>IoU (Intersection over Union):</strong> Predicted box ও ground truth box-এর overlap measure। IoU ≥ 0.5 হলে সাধারণত True Positive।</p>
    <p><strong>Precision-Recall Curve:</strong> Confidence threshold vary করলে precision ও recall বদলায়।</p>
    <p><strong>Average Precision (AP):</strong> একটি class-এর precision-recall curve-এর area under curve।</p>
    <p><strong>mAP (mean AP):</strong> সব class-এর AP-এর mean।</p>
    <ul>
      <li><strong>mAP@0.5:</strong> IoU threshold = 0.5 (PASCAL VOC standard)</li>
      <li><strong>mAP@0.5:0.95:</strong> IoU 0.5 থেকে 0.95 (step 0.05) এর mean — COCO standard, কঠিন</li>
    </ul>
    <pre><code>import numpy as np

def compute_ap(recalls, precisions):
    """
    Average Precision = area under precision-recall curve
    11-point interpolation method (PASCAL VOC)
    """
    ap = 0
    for threshold in np.arange(0, 1.1, 0.1):
        # এই recall threshold-এর উপরে সর্বোচ্চ precision
        precisions_at_recall = precisions[recalls >= threshold]
        max_precision = np.max(precisions_at_recall) if len(precisions_at_recall) > 0 else 0
        ap += max_precision / 11
    return ap

# Simulated detection results
recalls    = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
precisions = np.array([1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

ap = compute_ap(recalls, precisions)
print(f"Average Precision: {ap:.4f}")

# mAP across multiple classes
class_aps = {
    'person':    0.85,
    'car':       0.78,
    'bicycle':   0.62,
    'dog':       0.71,
    'cat':       0.69,
}

map_score = np.mean(list(class_aps.values()))
print(f"\nPer-class AP: {class_aps}")
print(f"mAP: {map_score:.4f}")
</code></pre>
  `,
};
