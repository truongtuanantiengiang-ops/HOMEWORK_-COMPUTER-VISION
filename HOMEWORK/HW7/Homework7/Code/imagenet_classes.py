import json
import urllib.request

def load_imagenet_classes():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    class_file = "imagenet_classes.txt"
    urllib.request.urlretrieve(url, class_file)

    with open(class_file) as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

IMAGENET_CLASSES = load_imagenet_classes()
