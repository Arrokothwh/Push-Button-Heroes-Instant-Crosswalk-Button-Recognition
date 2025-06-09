# Pedestrian Button Direction Classification

This repository presents a two-stage framework for pedestrian push-button detection and direction classification in urban street scenes. The project addresses challenges in small-object recognition, interpretability, and training stability in low-data regimes.

## 📌 Overview

We propose a pipeline consisting of:

1. **YOLOv8-based object detection** for localizing pedestrian push-buttons in street-level images.
2. **Image classifier (CNN or Transformer)** for predicting crossing direction ("left" or "right") using cropped button regions.

In addition, we explore the use of **synthetic arrow injection** and **feature attribution visualization (Grad-CAM, HiResCAM)** to guide the model to focus on semantically meaningful features.

---

## 🗂️ Dataset

Our custom dataset contains over 1,00 images collected from diverse intersections. Each sample includes:

- Bounding boxes for pedestrian button locations
- Direction labels: `"left"` or `"right"`

We also include synthetic arrow images to enhance generalization and interpretability.

> **Note:** Due to limited real-world availability, our test set is small (n=14). Evaluation results should be interpreted cautiously.

---

## 🧠 Model Architecture

### Stage 1: Detection
- **Backbone:** YOLOv8
- **Input:** Full-resolution street image
- **Output:** Bounding boxes for button regions

### Stage 2: Classification
- **Backbones:** ResNet-18, ResNet-34, TinyViT, ViT-B/16, DINOv2-Small
- **Input:** Cropped \(320 \times 320\) button images
- **Loss:** Binary Cross Entropy
- **Metrics:** Accuracy, Macro F1-Score

---

## 🔬 Key Findings

- Pretrained models consistently outperform randomly initialized models.
- Transformer backbones (especially DINOv2) achieve 100% test accuracy, but may overfit due to small test size.
- Grad-CAM and HiResCAM visualizations reveal that models sometimes focus on irrelevant cues (e.g. fingers), which motivates arrow injection.
- Synthetic arrows significantly improve attention alignment and convergence stability.

---

## 📊 Results Summary

| Model                             | Params (M) | Architecture        | Accuracy (%) | F1 Score (%) |
|----------------------------------|------------|---------------------|--------------|--------------|
| ResNet-18                        | 11.3       | CNN                 | 92.8         | 92.3         |
| ResNet-34                        | 21.4       | CNN                 | 94.2         | 92.3         |
| TinyViT                          | 20.7       | CNN + Transformer   | 96.3         | 83.3         |
| ViT-B/16                         | 85.5       | Transformer         | 92.8         | 92.3         |
| DINOv2 Small                     | 22.1       | Transformer         | 100.0        | 100.0        |
| TinyViT (Attention Correction)   | 20.7       | CNN + Transformer   | 92.3         | 91.0         |
| DINOv2 Small (Attention Correction) | 22.1    | Transformer         | 92.3         | 91.0         |

