# Car Parts Segmentation using YOLOv8

This project focuses on segmenting car parts in images using the YOLOv8 segmentation model. The goal is to accurately detect and segment various car parts, such as wheels, doors, mirrors, headlights, and bumpers, in images. This task is essential for applications like automotive repair, quality control, and autonomous driving.

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
The model is trained on a custom dataset containing images of car parts. The dataset is organized into training and validation sets, with annotations for each car part. The dataset configuration is defined in a YAML file, which specifies the paths to the images and the class names.

---

## Results
- **Training**: The model is trained for 30 epochs with an image size of 640x640.
- **Validation**: The model achieves high accuracy, as measured by mAP@0.5 and mAP@0.5:0.95.
- **Prediction**: The model can accurately segment car parts in new images, even under challenging conditions.

---

## Usage
To use this project, follow these steps:

1. **Install Dependencies**: Ensure the Ultralytics library is installed.
2. **Train the Model**: Train the YOLOv8 model on your custom dataset.
3. **Validate the Model**: Evaluate the model's performance on the validation set.
4. **Predict on New Images**: Use the trained model to segment car parts in new images.

---

## Folder Structure
The project is organized as follows:
```
carparts-segmentation/
├── carparts-seg.yaml          # Dataset configuration file
├── runs/                      # Output folder for training and prediction
│   ├── train-v1/              # Training outputs (weights, logs, etc.)
│   └── predict-v1/            # Prediction outputs (images, labels, etc.)
└── README.md                  # This file
```

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- Ultralytics for the YOLOv8 implementation.
- Google Colab for providing a free GPU environment.

