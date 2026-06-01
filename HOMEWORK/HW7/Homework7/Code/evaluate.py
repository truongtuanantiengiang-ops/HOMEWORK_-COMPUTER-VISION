import os
import json
import torch
import pandas as pd
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageDraw

# 1. Load AlexNet Pretrained
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
model.eval()

# 2. Chuẩn hóa ảnh
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# 3. Load danh sách class của ImageNet

imagenet_classes = models.AlexNet_Weights.IMAGENET1K_V1.meta["categories"]


# 4. Load class_map.json
with open("class_map.json", "r", encoding="utf-8") as f:
    class_map = json.load(f)

# Convert class names → index
name_to_idx = {name: i for i, name in enumerate(imagenet_classes)}

# Chuyển class_map (tên) → class_map_idx (index)
class_map_idx = {}
for folder_class, names in class_map.items():
    indices = []
    for n in names:
        if n in name_to_idx:
            indices.append(name_to_idx[n])
    class_map_idx[folder_class] = set(indices)

print("\nLoaded class_map.json:")
for k, v in class_map_idx.items():
    print(f" - {k}: {len(v)} ImageNet classes mapped")
print()


# Hàm dự đoán Top-5
def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    top5_prob, top5_idx = torch.topk(probs, 5)
    return [(int(idx), float(top5_prob[i])) for i, idx in enumerate(top5_idx)]

# Vẽ top-5 lên ảnh
def draw_top5(img_path, top5, save_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    y = 5
    for idx, prob in top5:
        cls_name = imagenet_classes[idx]
        draw.text((5, y), f"{cls_name}: {prob:.3f}", fill="red")
        y += 20

    img.save(save_path)

# 5. Đọc dataset và đánh giá
root = "dataset"
output_dir = "classification_results"
os.makedirs(output_dir, exist_ok=True)

classes = sorted(os.listdir(root))

# results[c] = [top1, top2, top3, top4, top5, total]
results = {c: [0, 0, 0, 0, 0, 0] for c in classes}

for c in classes:
    class_path = os.path.join(root, c)
    imgs = sorted(os.listdir(class_path))

    valid_indices = class_map_idx[c]

    print(f"\nProcessing class '{c}' with {len(imgs)} images...")

    for img_name in imgs:
        img_path = os.path.join(class_path, img_name)

        # --- Dự đoán ---
        top5 = predict(img_path)

        # --- Vẽ hình ---
        save_path = f"{output_dir}/{c}_{img_name}"
        draw_top5(img_path, top5, save_path)

        # Chuyển top5 (index) thành list indices
        top5_indices = [t[0] for t in top5]

        # --- Tính thống kê ---
        results[c][5] += 1  # total++

        if top5_indices[0] in valid_indices: results[c][0] += 1
        if any(i in valid_indices for i in top5_indices[:2]): results[c][1] += 1
        if any(i in valid_indices for i in top5_indices[:3]): results[c][2] += 1
        if any(i in valid_indices for i in top5_indices[:4]): results[c][3] += 1
        if any(i in valid_indices for i in top5_indices[:5]): results[c][4] += 1

# 6. Xuất Excel

df = pd.DataFrame(results, index=["top1", "top2", "top3", "top4", "top5", "total"]).T

# Convert accuracy %
for col in ["top1", "top2", "top3", "top4", "top5"]:
    df[col] = (df[col] / df["total"]) * 100

df.to_excel("evaluation_results.xlsx")

print(" - evaluation_results.xlsx (bảng kết quả chính xác)")
