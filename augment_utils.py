import os
import cv2
import numpy as np
import hashlib
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime



INPUT_DIR = None
OUTPUT_DIR = None


# Helper functions
def load_image_and_boxes(filename):
    img_path = os.path.join(INPUT_DIR, "images", filename + ".jpg")
    txt_path = os.path.join(INPUT_DIR, "labels", filename + ".txt")

    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    boxes = []
    class_labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            # 转为 [x_min, y_min, x_max, y_max]
            xmin = (x - w / 2) * width
            ymin = (y - h / 2) * height
            xmax = (x + w / 2) * width
            ymax = (y + h / 2) * height
            boxes.append([xmin, ymin, xmax, ymax])
            class_labels.append(int(cls))
    return image, boxes, class_labels, width, height

def save_augmented(image, boxes, class_labels, filename):
    hash_code = hashlib.md5(image.tobytes()).hexdigest()[:4]
    now = datetime.now()
    timestamp = now.strftime("%m%d%H%M%S")
    new_filename = f"{os.path.splitext(filename)[0]}-{hash_code}-{timestamp}"

    image_out_path = os.path.join(OUTPUT_DIR, 'images', new_filename + ".jpg")
    label_out_path = os.path.join(OUTPUT_DIR, 'labels', new_filename + ".txt")

    os.makedirs(os.path.dirname(image_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(label_out_path), exist_ok=True)

    height, width = image.shape[:2]
    cv2.imwrite(image_out_path, image)

    with open(label_out_path, 'w') as f:
        for cls, box in zip(class_labels, boxes):
            x_min, y_min, x_max, y_max = box
            # 转为YOLO格式
            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


# Add Gaussian noise
def add_gaussian_noise(filename):
    image, boxes, class_labels, _, _ = load_image_and_boxes(filename)

    noisy = image + np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    save_augmented(noisy, boxes, class_labels, filename)


# Adj Brightness
def adjust_random_brightness(filename):
    image, boxes, class_labels, _, _ = load_image_and_boxes(filename)

    # 双峰正态分布：50% 概率取 0.4，50% 概率取 1.8
    if random.random() < 0.5:
        factor = np.random.normal(loc=0.4, scale=0.1)
    else:
        factor = np.random.normal(loc=1.8, scale=0.1)

    # 限制亮度范围在合理区间
    factor = max(0.1, min(2.5, factor))

    bright = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    save_augmented(bright, boxes, class_labels, filename)


# Block out random chunk

def add_black_rect(filename):
    image, boxes, class_labels, width, height = load_image_and_boxes(filename)

    max_attempts = 20
    for _ in range(max_attempts):
        rect_w = random.randint(int(0.3 * width), int(0.5 * width))
        rect_h = random.randint(int(0.3 * height), int(0.5 * height))
        x1 = random.randint(0, width - rect_w)
        y1 = random.randint(0, height - rect_h)
        x2 = x1 + rect_w
        y2 = y1 + rect_h

        overlap = False
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if not (x2 < xmin or x1 > xmax or y2 < ymin or y1 > ymax):
                overlap = True
                break
        if not overlap:
            image[y1:y2, x1:x2] = 0
            break

    save_augmented(image, boxes, class_labels, filename)


# Horizontal flip

def horizontal_flip(filename):
    image, boxes, class_labels, w, h = load_image_and_boxes(filename)

    transform = A.Compose([
        A.HorizontalFlip(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    augmented = transform(image=image, bboxes=boxes, class_labels=class_labels)

    save_augmented(augmented['image'], augmented['bboxes'], augmented['class_labels'], filename)


# random rotate
def random_rotate(filename):
    image, boxes, class_labels, w, h = load_image_and_boxes(filename)

    transform = A.Compose([
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    augmented = transform(image=image, bboxes=boxes, class_labels=class_labels)

    save_augmented(augmented['image'], augmented['bboxes'], augmented['class_labels'], filename)


# random scale
def random_scale_with_padding(filename):
    image, boxes, class_labels, w, h = load_image_and_boxes(filename)

    # 双峰正态分布采样 scale
    if random.random() < 0.5:
        scale_factor = np.random.normal(loc=0.5, scale=0.1)
    else:
        scale_factor = np.random.normal(loc=1.8, scale=0.1)

    # 限制 scale 在合理区间
    scale_factor = max(0.1, min(2.5, scale_factor))

    transform = A.Compose([
        A.Affine(scale=scale_factor, fit_output=True, mode=cv2.BORDER_CONSTANT, cval=0),
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    augmented = transform(image=image, bboxes=boxes, class_labels=class_labels)

    save_augmented(augmented['image'], augmented['bboxes'], augmented['class_labels'], filename)
# def random_scale_with_padding(filename):
#     image, boxes, class_labels, w, h = load_image_and_boxes(filename)

#     transform = A.Compose([
#         A.RandomScale(scale_limit=(0.5, 1.5), p=1.0),
#         A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
#     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

#     augmented = transform(image=image, bboxes=boxes, class_labels=class_labels)

#     save_augmented(augmented['image'], augmented['bboxes'], augmented['class_labels'], filename)
