from PIL import Image, ImageFilter, ImageDraw
import os

input_dir = "dataset"
output_dir = "distorted_dataset"
os.makedirs(output_dir, exist_ok=True)


def distort(img_path):
    img = Image.open(img_path).convert("RGB")

    blur = img.filter(ImageFilter.GaussianBlur(8))

    distort_img = img.resize((300, 150))

    block = img.copy()
    draw = ImageDraw.Draw(block)
    draw.rectangle((50, 50, 150, 150), fill="black")

    return blur, distort_img, block


for cls in os.listdir(input_dir):
    os.makedirs(f"{output_dir}/{cls}", exist_ok=True)

    for img_name in os.listdir(f"{input_dir}/{cls}"):
        img_path = f"{input_dir}/{cls}/{img_name}"

        blur, distort_img, block = distort(img_path)

        blur.save(f"{output_dir}/{cls}/blur_{img_name}")
        distort_img.save(f"{output_dir}/{cls}/distort_{img_name}")
        block.save(f"{output_dir}/{cls}/block_{img_name}")

print("Generated distorted images!")
