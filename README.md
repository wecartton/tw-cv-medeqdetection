# Medical Equipment Detection & Inventory Automation

**Computer Vision pipeline utilizing YOLOv5 and Roboflow to automate hospital equipment tracking.**

## Project Overview
This project presents an automated computer vision system designed to detect and classify 18 distinct medical devices in healthcare environments. By treating object detection as a scalable engineering problem, the project focuses heavily on data curation, preprocessing pipelines, and robust evaluation to address the critical need for efficient medical tool management.

## The Business Problem
In complex hospital environments with hundreds of devices, manual equipment tracking is inherently slow and prone to human error. During critical moments, the inability to immediately locate specific surgical or diagnostic tools poses severe risks to patient safety. This project serves as a proof-of-concept for a camera-based, automated inventory tracking system to replace manual logging.

## Data Engineering Pipeline
A significant portion of this project was dedicated to building a high-quality dataset from scratch, proving that data quality dictates model performance. 

**1. Data Curation**
The initial dataset consisted of only 235 raw images, sourced manually through search engines and mobile phone cameras. Roboflow was utilized as the primary platform for efficient bounding-box annotation and pipeline management.

**2. Preprocessing Strategy**
To ensure the Convolutional Neural Network (CNN) learns optimal features without spatial bias, rigorous standardization was applied:
* **Auto-Orient:** Stripped EXIF data to ensure consistent image direction.
* **Auto-Adjust Contrast:** Applied Histogram Equalization to normalize varied hospital lighting conditions.
* **Center Cropping (640x640):** Maintained a 1:1 square aspect ratio to prevent image distortion, ensuring the model maintains low bias toward object orientation.

**3. Data Augmentation**
To overcome the limitations of a small dataset and prevent model overfitting, geometric transformations were applied to scale the dataset by 5x, resulting in a final training set of 1,019 images.
* **Flipping:** Horizontal and Vertical.
* **Rotation:** 90-degree Clockwise, Counter-Clockwise, and Upside Down.

## Model Architecture
The system utilizes the **YOLOv5** architecture. YOLO (You Only Look Once) was selected over two-stage detectors (like Faster R-CNN) due to its optimal trade-off between real-time inference speed and detection accuracy, which is crucial for a deployable healthcare API.

## Results & Evaluation Metrics
The model was evaluated based on its ability to accurately bound and classify the 18 device classes.
* **Precision:** 91.7% (The model is highly accurate when it makes a positive prediction).
* **mAP (Mean Average Precision):** 46.3%
* **Recall:** 27.0%

## Engineering Insights & Error Analysis
A critical part of deploying AI models is understanding their failure modes. Post-training evaluation revealed a relatively low recall rate, driven by a high number of False Negatives in specific classes.

* **The Problem:** The model struggled significantly to distinguish between visually similar tools, most notably *forceps*, *clamps*, and parts of the *hiography machine*.
* **The Root Cause:** These tools share nearly identical metallic textures and geometric shapes, differing only in micro-features (e.g., the locking mechanism or tip serrations).
* **The Solution (Next Iteration):** Algorithmic tweaking or hyperparameter tuning is insufficient to solve this. The engineering roadmap requires a data-centric approach: implementing fine-grained feature annotation that explicitly focuses on the distinguishing mechanical parts of these tools, rather than bounding the entire object.

## Tech Stack
* **Language:** Python
* **Computer Vision:** YOLOv5, Convolutional Neural Networks (CNN)
* **Data Pipeline:** Roboflow (Annotation, Preprocessing, Augmentation)
* **Evaluation:** Precision, Recall, mAP evaluation metrics
