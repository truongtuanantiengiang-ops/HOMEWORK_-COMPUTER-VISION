import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import json

DATASET_DIR = "dataset"
SAVE_DIR = "classification_results"

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load Imagenet class names
with open("imagenet_class_index.json") as f:
    idx2label = {int(k): v[1] for k, v in json.load(f).items()}


def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        prob = torch.nn.functional.softmax(out, dim=1)

    top5_prob, top5_catid = torch.topk(prob, 5)

    top5_prob = top5_prob.cpu().numpy()[0]
    top5_catid = top5_catid.cpu().numpy()[0]

    result_top5 = [(idx2label[top5_catid[i]], float(top5_prob[i]))
                   for i in range(5)]

    return img, result_top5


def plot_class_results(class_name, image_paths):
    """
    Create 2 summary images:
    (1) Classification Results 5x4
    (2) Top-5 Predictions 5x4
    """

    #  Classification Results
    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    fig.suptitle(f"Classification Results - {class_name.upper()}", fontsize=20)

    for ax, img_path in zip(axes.flatten(), image_paths):
        img, top5 = predict(img_path)
        top1 = top5[0]

        ax.imshow(img)
        ax.axis("off")

        label = f"{top1[0]}\n{top1[1]:.3f}"
        ax.set_title(label,
                     fontsize=9,
                     color="green" if class_name in top1[0] else "red")

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"{class_name}_result.png"))
    plt.close()

    #  Top-5 Predictions Summary
    fig2, axes2 = plt.subplots(4, 5, figsize=(18, 14))
    fig2.suptitle(f"Top 5 Predicted Classes - {class_name.upper()}",
                  fontsize=20)

    for ax, img_path in zip(axes2.flatten(), image_paths):
        img, top5 = predict(img_path)

        text = "\n".join([f"{lbl}: {p:.3f}" for lbl, p in top5])
        ax.text(0.0, 0.5, text, fontsize=7)
        ax.axis("off")

    fig2.tight_layout()
    fig2.savefig(os.path.join(SAVE_DIR, f"{class_name}_top5.png"))
    plt.close()


def run_all():
    classes = os.listdir(DATASET_DIR)

    for cls in classes:
        class_path = os.path.join(DATASET_DIR, cls)
        image_paths = [
            os.path.join(class_path, f) for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        image_paths = image_paths[:20]  # exactly 20 images

        print(f"Processing class: {cls} ({len(image_paths)} images)")
        plot_class_results(cls, image_paths)

    print("\nDONE! Results saved in:", SAVE_DIR)

if __name__ == "__main__":
    run_all()
