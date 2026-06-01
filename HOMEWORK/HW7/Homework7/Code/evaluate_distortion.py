import os
import json
import torch
import pandas as pd
from torchvision import models, transforms
from PIL import Image

# =========================
# Load ImageNet class names
from imagenet_classes import IMAGENET_CLASSES


# =========================
# Load AlexNet pretrained
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# =========================
# Load class_map.json
# =========================
with open("class_map.json", "r") as f:
    CLASS_MAP = json.load(f)      # {"cat":[281,282,283,...], "dog":[207,...]}


# =========================
# Preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================
# Prediction (top-5)
# =========================
def predict_top5(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        prob = torch.softmax(outputs, dim=1)[0]

    top5_prob, top5_idx = torch.topk(prob, 5)
    return top5_idx.cpu().numpy(), top5_prob.cpu().numpy()


# =========================
# Evaluate dataset
# =========================
def evaluate_dataset(dataset_path):
    classes = sorted(os.listdir(dataset_path))
    results = {c: {"top1": 0, "top2": 0, "top3": 0, "top4": 0, "top5": 0, "total": 0}
               for c in classes}

    for c in classes:
        class_dir = os.path.join(dataset_path, c)

        # Only image files
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        valid_idx = set(CLASS_MAP[c])  # ImageNet indices for this class

        for img_file in images:
            img_path = os.path.join(class_dir, img_file)

            top5_idx, _ = predict_top5(img_path)

            results[c]["total"] += 1

            # Top-k evaluation
            if top5_idx[0] in valid_idx: results[c]["top1"] += 1
            if any(i in valid_idx for i in top5_idx[:2]): results[c]["top2"] += 1
            if any(i in valid_idx for i in top5_idx[:3]): results[c]["top3"] += 1
            if any(i in valid_idx for i in top5_idx[:4]): results[c]["top4"] += 1
            if any(i in valid_idx for i in top5_idx[:5]): results[c]["top5"] += 1

    # Convert to DataFrame
    df = pd.DataFrame(results).T
    for col in ["top1", "top2", "top3", "top4", "top5"]:
        df[col] = df[col] / df["total"] * 100

    return df


# =========================
# Run both datasets
# =========================
normal_df = evaluate_dataset("dataset")
distorted_df = evaluate_dataset("distorted_dataset")

# =========================
# Compute impact (% change)
# =========================
impact_df = distorted_df - normal_df


# =========================
# Save Excel with 3 sheets
# =========================
with pd.ExcelWriter("distortion_evaluation.xlsx") as writer:
    normal_df.to_excel(writer, sheet_name="Normal_Results")
    distorted_df.to_excel(writer, sheet_name="Distorted_Results")
    impact_df.to_excel(writer, sheet_name="Impact")

print("DONE! Saved: distortion_evaluation.xlsx")
