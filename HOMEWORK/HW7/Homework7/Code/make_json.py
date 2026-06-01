import json
from torchvision.models import AlexNet_Weights

classes = AlexNet_Weights.IMAGENET1K_V1.meta["categories"]

imagenet_index = {str(i): ["n" + str(i).zfill(8), classes[i]] for i in range(1000)}

with open("imagenet_class_index.json", "w") as f:
    json.dump(imagenet_index, f, indent=4)

print("File saved: imagenet_class_index.json")