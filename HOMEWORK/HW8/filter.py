import random
import os
import shutil
import json
from pycocotools.coco import COCO

# =============================== CONFIG ===============================
ANN_FILE = r'D:\Automation\Thị giác máy\HW\HW8\annotations_trainval2017\annotations\instances_val2017.json'
IMAGE_DIR = r'D:\Automation\Thị giác máy\HW\HW8\val2017\val2017'   # thư mục chứa 5000 ảnh COCO
OUTPUT_DIR = r'D:\Automation\Thị giác máy\HW\HW8\100_images'       # thư mục ảnh lọc ra
OUTPUT_ANN_FILE = r'D:\Automation\Thị giác máy\HW\HW8\100_images_annotations.json'

NUM_CLASSES = 5
IMAGES_PER_CLASS = 20

TARGET_CLASSES = ['person', 'cat', 'dog', 'cow', 'giraffe']
# =====================================================================


def filter_and_create_json():
    # ----------- 1. Load COCO annotation -----------
    coco = COCO(ANN_FILE)

    cat_ids = coco.getCatIds(catNms=TARGET_CLASSES)
    if len(cat_ids) != NUM_CLASSES:
        print("❌ Không tìm thấy đủ class. Kiểm tra TARGET_CLASSES!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selected_filenames = set()
    selected_image_ids = set()

    # ----------- 2. Lọc ảnh theo từng class -----------
    for cat_id, cat_name in zip(cat_ids, TARGET_CLASSES):
        print(f"\n--- Class: {cat_name} ---")

        img_ids_all = coco.getImgIds(catIds=[cat_id])

        if len(img_ids_all) < IMAGES_PER_CLASS:
            sample_ids = img_ids_all
        else:
            sample_ids = random.sample(img_ids_all, IMAGES_PER_CLASS)

        imgs_info = coco.loadImgs(sample_ids)

        # Folder theo class
        class_out_dir = os.path.join(OUTPUT_DIR, cat_name.replace(" ", "_"))
        os.makedirs(class_out_dir, exist_ok=True)

        # Copy ảnh
        for info in imgs_info:
            fname = info['file_name']
            src = os.path.join(IMAGE_DIR, fname)
            dst = os.path.join(class_out_dir, fname)

            if os.path.exists(src):
                shutil.copyfile(src, dst)
                selected_filenames.add(fname)
                selected_image_ids.add(info['id'])
            else:
                print("⚠️ File missing:", src)

        print(f"✓ Copied {len(imgs_info)} images for class {cat_name}")

    # ----------- 3. Load JSON COCO gốc -----------
    with open(ANN_FILE, "r") as f:
        coco_json = json.load(f)

    # ----------- 4. Lọc IMAGE ----------
    new_images = [img for img in coco_json["images"] if img["file_name"] in selected_filenames]

    # ----------- 5. Lọc ANNOTATION ----------
    new_annotations = [
        ann for ann in coco_json["annotations"]
        if ann["image_id"] in selected_image_ids and ann["category_id"] in cat_ids
    ]

    # ----------- 6. Lọc CATEGORY ----------
    new_categories = [c for c in coco_json["categories"] if c["id"] in cat_ids]

    # ----------- 7. Xuất ra JSON mới ----------
    new_coco = {
        "info": coco_json.get("info", {}),
        "licenses": coco_json.get("licenses", []),
        "categories": new_categories,
        "images": new_images,
        "annotations": new_annotations
    }

    with open(OUTPUT_ANN_FILE, "w") as f:
        json.dump(new_coco, f)

    # ----------- 8. Thông báo kết quả ----------
    print("\n================= DONE =================")
    print("Created:", OUTPUT_ANN_FILE)
    print("Images:", len(new_images))
    print("Annotations:", len(new_annotations))
    print("Categories:", len(new_categories))
    print("========================================")


if __name__ == "__main__":
    filter_and_create_json()
