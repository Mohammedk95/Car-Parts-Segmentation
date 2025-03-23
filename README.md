# Car Parts Segmentation using YOLOv8

This project focuses on segmenting car parts in images using the YOLOv8 segmentation model. The goal is to accurately detect and segment various car parts, such as wheels, doors, mirrors, headlights, bumpers, and more, in images. This task is essential for applications like automotive repair, quality control, and autonomous driving.

---

## Problem Statement
Segmenting car parts in images is a challenging task due to factors like varying lighting conditions, occlusions, and complex backgrounds. Traditional methods often struggle to handle these complexities, leading to inaccurate results. This project addresses these challenges by leveraging the power of YOLOv8, a state-of-the-art deep learning model, to achieve precise and reliable car parts segmentation.

---

## Solution
The solution involves training a YOLOv8 segmentation model on a custom dataset of car parts. The trained model is then used to predict and segment car parts in new, unseen images. The process includes:

1. **Training**: Fine-tuning the YOLOv8 model on a custom dataset.
2. **Validation**: Evaluating the model's performance using metrics like mAP (mean Average Precision).
3. **Prediction**: Using the trained model to segment car parts in new images.

---

## Dataset
- The model is trained on a custom dataset containing images of car parts. The dataset is organized into training and validation sets, with annotations for each car part. The dataset configuration is defined in a YAML file, which specifies the paths to the images and the class names.
- Dataset link: https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm/dataset/4

---

## Results
- **Training**: The model is trained for **60 epochs** with an image size of 640x640 using **Google Colab Pro with an A100 GPU** for accelerated training.
- **Validation**: The model achieves high accuracy, as measured by mAP@0.5 and mAP@0.5:0.95.
- **Prediction**: The model can accurately segment car parts in new images, even under challenging conditions.
- ![Screenshot 2025-03-23 061018](https://github.com/user-attachments/assets/a06c278b-b87c-46fb-92b5-4616d311fe2b)
- ![Screenshot 2025-03-23 061127](https://github.com/user-attachments/assets/ce630e05-ef4e-48bb-8f65-96b2ca144396)
- ![Screenshot 2025-03-23 061143](https://github.com/user-attachments/assets/ad52e4c4-b729-4ca5-8b2a-9c9933f51818)
- ![Screenshot 2025-03-23 061225](https://github.com/user-attachments/assets/b803ff3a-1a22-4bc9-87b8-50518377cc5f)

---

## Google Colab Notebook
You can access and run the full code in the Google Colab notebook below. The notebook includes all the steps, from installing dependencies to training, validation, and prediction.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sJoWCRZ06s4Gno-pHOhQi0l03Ffu1Y9x?usp=sharing)

---

## Usage
To use this project, follow these steps:

1. **Install Dependencies**: Ensure the Ultralytics library is installed.
2. **Train the Model**: Train the YOLOv8 model on your custom dataset.
3. **Validate the Model**: Evaluate the model's performance on the validation set.
4. **Predict on New Images**: Use the trained model to segment car parts in new images.

---

## Notes
- **Prediction Results**: The output of the prediction step (e.g., segmented images) is **only for visualization purposes**. It helps in understanding the model's performance and does not represent any quantitative evaluation.
- **Hardware**: This project utilizes **Google Colab Pro with an A100 GPU** for faster training and inference.

---

## Acknowledgments
- Reference: [Ultralytics](https://docs.ultralytics.com/datasets/segment/carparts-seg/) for the YOLOv8 implementation and dataset reference.
- Google Colab Pro for providing access to high-performance A100 GPUs.

---
