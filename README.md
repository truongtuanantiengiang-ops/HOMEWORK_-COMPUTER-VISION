# Machine Vision (EE3077) - Homework Reports

## 📌 Introduction
This repository contains the source code and homework reports for the **Machine Vision** course (Course Code: EE3077), offered by the Department of Automatic Control, Faculty of Electrical and Electronics Engineering at Ho Chi Minh City University of Technology (HCMUT)[cite: 1]. 

The report was completed in December 2025 under the guidance of Dr. Phạm Việt Cường[cite: 1].

## 👥 Team Members
Project executed by students from Class L01[cite: 1]:

| Full Name | Student ID |
| :--- | :--- |
| Nguyễn Thiện Khả | 2211543[cite: 1] |
| Nguyễn Thái Tuấn | 2213759[cite: 1] |
| Trương Tuấn An | 2210041[cite: 1] |
| Trương Trọng Nghĩa | 2212245[cite: 1] |
| Trần Đình Nhật Huy | 2211279[cite: 1] |
| Lê Trường Sơn | 1813847[cite: 1] |

## 📚 Homework Breakdown

* **HW1: Morphological Operations**
  * Processing and restoring missing black areas inside coins using Dilation and Erosion operations[cite: 1].
* **HW2: Image Segmentation**
  * Reprogramming and comparing image segmentation thresholding algorithms: Basic global thresholding (heuristic), Otsu's method, and Adaptive thresholding (adaptthresh) using MATLAB[cite: 1].
* **HW3: String State Detection**
  * Determining the state/orientation of a string (vertical, horizontal, inclined) using Canny edge detection and the Hough Transform[cite: 1].
* **HW4: Object Detector Training**
  * Training a custom object detection model (similar to "Train Stop Sign Detector") to detect faces[cite: 1]. 
  * Due to limited data, the Cascade Object Detector was trained using 300 positive images and 1349 negative images across 12 stages[cite: 1].
* **HW5: Face Detection Evaluation**
  * Utilizing a pre-built MATLAB model to perform face detection on 20 images (each containing 5 target objects)[cite: 1].
  * Evaluating model performance metrics based on the confusion matrix: Precision (0.8812), Recall (0.89), and Accuracy (0.7946)[cite: 1].
* **HW6: AlexNet Architecture Analysis**
  * Inspecting and calculating the number of trainable parameters for each layer in the AlexNet model. The total number of parameters across 5 Convolutional layers and 3 Fully-Connected layers is calculated to be 62,378,344[cite: 1].
* **HW7: AlexNet Performance on Custom Dataset**
  * Evaluating the performance of the AlexNet model (pretrained on ImageNet, PyTorch framework) on a custom dataset of 5 classes (butterfly, cat, dog, horse, spider), with 20 images per class[cite: 1].
  * Analyzing the impact of image transformations—Gaussian blur, Affine distortion, and random Blackout—on the model's accuracy (Top-1 and Top-5 error rates)[cite: 1].
* **HW8: YOLOv8n Model Evaluation**
  * Testing the lightweight YOLOv8n object detection model on a 100-image dataset containing 5 target classes: person, cat, dog, cow, and giraffe[cite: 1].
  * Visualizing predictions (Bounding Boxes, Scores) and evaluating overall performance using Mean Average Precision (mAP) under the COCO evaluation protocol, achieving an mAP of 0.4869[cite: 1].

## 🛠 Tools & Technologies
* **Programming Languages**: MATLAB, Python[cite: 1].
* **Frameworks/Libraries**: PyTorch[cite: 1].
* **Models Utilized**: Cascade Object Detector, AlexNet, YOLOv8n[cite: 1].

## 🔗 References & Source Code
* **Source Code Repository**: [GitHub Link](https://github.com/truongtuanantiengiang-ops/HOMEWORK)[cite: 1].
* **Datasets Used**: COCO 2017 annotations (for YOLO), Kaggle (for AlexNet custom dataset)[cite: 1].
