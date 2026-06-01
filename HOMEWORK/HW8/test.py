from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

# ---------------- CONFIG ----------------
IMAGE_DIR = r"D:\Automation\Thị giác máy\HW\HW8\100_images"
ANN_FILE  = r"D:\Automation\Thị giác máy\HW\HW8\100_images_annotations.json"
YOLO_MODEL = "yolov8n.pt"  # hoặc yolov8s.pt
COLS = 5  # số cột khi hiển thị gallery
# ----------------------------------------

# 1. Load YOLO model
model = YOLO(YOLO_MODEL)

# 2. Load COCO annotation
coco_gt = COCO(ANN_FILE)

# Lấy danh sách 5 class trong annotation
target_cats = coco_gt.loadCats(coco_gt.getCatIds())
cat_name_to_id = {c["name"]: c["id"] for c in target_cats}
TARGET_CLASSES = list(cat_name_to_id.keys())  # ['person', 'cat', 'dog', 'cow', 'giraffe']

# 3. Lấy danh sách tất cả ảnh và image_id
image_files = []
image_ids = []
for root, dirs, files in os.walk(IMAGE_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(root, f)
            image_files.append(full_path)
            img_id_list = coco_gt.getImgIds()
            for img in coco_gt.loadImgs(img_id_list):
                if img["file_name"] == os.path.basename(full_path):
                    image_ids.append(img["id"])
                    break

print("Số ảnh tìm thấy:", len(image_files))

# --- 4. Detect và chuẩn bị COCO prediction ---
coco_pred_list = []
for i, img_path in enumerate(image_files):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)

    boxes = results[0].boxes.xyxy.cpu().numpy()    
    scores = results[0].boxes.conf.cpu().numpy()   
    labels_idx = results[0].boxes.cls.cpu().numpy()  
    labels = [model.names[int(idx)] for idx in labels_idx]

    for box, score, label in zip(boxes, scores, labels):
        if label not in TARGET_CLASSES:
            continue  # chỉ lấy 5 class
        x1, y1, x2, y2 = box.astype(int)
        coco_pred_list.append({
            "image_id": image_ids[i],
            "category_id": cat_name_to_id[label],
            "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            "score": float(score)
        })

# --- 5. Lưu prediction tạm và dùng COCOeval ---
pred_file = "temp_predictions.json"
with open(pred_file, "w") as f:
    json.dump(coco_pred_list, f)

coco_dt = coco_gt.loadRes(pred_file)
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()

# --- 6. Vẽ Precision-Recall cho từng class và tính AP ---
plt.figure(figsize=(10, 6))
for cat_name, cat_id in cat_name_to_id.items():
    precisions = coco_eval.eval['precision']  # shape = [TxRxKxAxM]
    cat_idx = list(coco_gt.getCatIds()).index(cat_id)
    
    # Lấy precision theo recall, trung bình qua các IoU
    precision = precisions[:, :, cat_idx, 0, -1]  # T x R
    precision_mean = np.mean(precision, axis=0)
    
    recall_levels = np.linspace(0, 1, precision_mean.shape[0])
    plt.plot(recall_levels, precision_mean, label=f"{cat_name}")

    # Tính AP cho class này (mean over all recall)
    ap = np.mean(precision_mean)
    print(f"AP cho class {cat_name}: {ap:.4f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve per class")
plt.grid(True)
plt.legend()
plt.show()

# --- Cleanup ---
os.remove(pred_file)

# --- 7. Hiển thị Gallery Toàn Bộ Ảnh với Bounding Box và Score ---
# Lấy tổng số ảnh
total_images = len(image_files)
# Kích thước gallery: 5 cột (COLS đã định nghĩa là 5)
IMAGES_PER_FIGURE = 10 # Số ảnh tối đa trong mỗi figure
COLS = 5
ROWS = IMAGES_PER_FIGURE // COLS # 2 hàng

# Tính số lượng figure cần thiết
num_figures = (total_images + IMAGES_PER_FIGURE - 1) // IMAGES_PER_FIGURE

print(f"Tổng số ảnh: {total_images}. Sẽ hiển thị trong {num_figures} figure (10 ảnh/figure).")

for figure_idx in range(num_figures):
    # Xác định chỉ số bắt đầu và kết thúc cho figure hiện tại
    start_index = figure_idx * IMAGES_PER_FIGURE
    end_index = min((figure_idx + 1) * IMAGES_PER_FIGURE, total_images)
    
    # Lấy danh sách file và ID cho figure hiện tại
    display_files = image_files[start_index:end_index]
    
    # Tạo Figure mới
    plt.figure(figsize=(20, 4 * ROWS)) # Giữ kích thước figure cố định
    plt.suptitle(f"Detection Results Gallery - Part {figure_idx + 1}/{num_figures} (Images {start_index+1} to {end_index})", fontsize=16)

    for i, img_path in enumerate(display_files):
        # Đọc và xử lý ảnh
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Lấy kết quả dự đoán (có thể bỏ qua bước này nếu đã lưu kết quả trước đó)
        results = model(img_rgb)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        labels_idx = results[0].boxes.cls.cpu().numpy()
        labels = [model.names[int(idx)] for idx in labels_idx]
        
        # Vẽ lên ảnh
        img_with_boxes = np.copy(img_rgb)
        
        # Map nhãn YOLO sang màu sắc
        colors = plt.cm.get_cmap('hsv', len(model.names)) 
        
        for box, score, label in zip(boxes, scores, labels):
            # Chỉ hiển thị 5 class mục tiêu
            if label not in TARGET_CLASSES:
                continue
                
            x1, y1, x2, y2 = box.astype(int)
            
            # Chọn màu dựa trên index của nhãn trong tất cả các nhãn YOLO
            try:
                 label_id_yolo = list(model.names.keys())[list(model.names.values()).index(label)]
            except ValueError:
                 # Trường hợp hiếm: label không có trong model.names, bỏ qua hoặc gán màu mặc định
                 continue 
                 
            color = colors(label_id_yolo / len(model.names))[:3]
            color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Vẽ Bounding Box
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # Chuẩn bị text: Label + Score
            text = f"{label}: {score:.2f}"
            
            # Vẽ nền cho text
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # Điều chỉnh vị trí y cho hộp văn bản để nó nằm trong ranh giới ảnh nếu cần
            text_y_pos = y1 - baseline
            cv2.rectangle(img_with_boxes, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
            
            # Đặt Text
            cv2.putText(img_with_boxes, text, (x1, text_y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Hiển thị vào subplot trong figure hiện tại
        plt.subplot(ROWS, COLS, i + 1)
        plt.imshow(img_with_boxes)
        plt.title(os.path.basename(img_path), fontsize=8) # Giảm cỡ chữ tiêu đề ảnh
        plt.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Điều chỉnh layout để có chỗ cho suptitle
    plt.show()

