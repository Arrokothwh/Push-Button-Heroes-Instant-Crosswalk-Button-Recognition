import os
import cv2
import numpy as np
import albumentations as A
import hashlib
import random
from datetime import datetime


# 由外部笔记本在运行时赋值
INPUT_DIR = None
OUTPUT_DIR = None

# ---------- I/O helpers ---------- #
def load_image_and_boxes(filename):
    """读取一张图像并返回 (image, boxes, class_labels, w, h)."""
    img_path = os.path.join(INPUT_DIR, "images", filename + ".jpg")
    txt_path = os.path.join(INPUT_DIR, "labels", filename + ".txt")

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(img_path)
    h, w = image.shape[:2]

    boxes, class_labels = [], []
    with open(txt_path, "r") as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.strip().split())
            xmin = (x - bw / 2) * w
            ymin = (y - bh / 2) * h
            xmax = (x + bw / 2) * w
            ymax = (y + bh / 2) * h
            boxes.append([xmin, ymin, xmax, ymax])
            class_labels.append(int(cls))
    return image, boxes, class_labels, w, h


def save_augmented(image, boxes, class_labels, filename):
    """保存增强后图像及 label，文件名格式: 原名-{hash}{时间戳}.ext"""
    # hash + 时间戳
    h4 = hashlib.md5(image.tobytes()).hexdigest()[:4]
    ts = datetime.now().strftime("%m%d%H%M%S")
    new_name = f"{os.path.splitext(filename)[0]}-{h4}{ts}"

    img_out = os.path.join(OUTPUT_DIR, "images", new_name + ".jpg")
    txt_out = os.path.join(OUTPUT_DIR, "labels", new_name + ".txt")

    os.makedirs(os.path.dirname(img_out), exist_ok=True)
    os.makedirs(os.path.dirname(txt_out), exist_ok=True)
    cv2.imwrite(img_out, image)

    h, w = image.shape[:2]
    with open(txt_out, "w") as f:
        for cls, (xmin, ymin, xmax, ymax) in zip(class_labels, boxes):
            x = ((xmin + xmax) / 2) / w
            y = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

# ---------- augment functions (chain-style) ---------- #
def add_gaussian_noise(image, boxes, class_labels):
    noisy = image + np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, boxes, class_labels


def adjust_random_brightness(image, boxes, class_labels):
    # 双峰分布
    factor = np.random.normal(0.7, 0.2) if random.random() < 0.5 else np.random.normal(1.6, 0.2)
    factor = max(0.3, min(2.0, factor))
    bright = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return bright, boxes, class_labels


def add_black_rect(image, boxes, class_labels):
    h, w = image.shape[:2]
    for _ in range(20):
        rw = random.randint(int(0.5 * w), int(0.7 * w))
        rh = random.randint(int(0.5 * h), int(0.7 * h))
        x1 = random.randint(0, w - rw)
        y1 = random.randint(0, h - rh)
        x2, y2 = x1 + rw, y1 + rh

        if all(x2 < bx[0] or x1 > bx[2] or y2 < bx[1] or y1 > bx[3] for bx in boxes):
            image[y1:y2, x1:x2] = 0
            break
    return image, boxes, class_labels



def horizontal_flip(image, boxes, class_labels):
    h, w = image.shape[:2]
    boxes_flipped = [[w - bx[2], bx[1], w - bx[0], bx[3]] for bx in boxes]
    flipped = cv2.flip(image, 1)
    return flipped, boxes_flipped, class_labels


def random_rotate(image, boxes, class_labels):
    transform = A.Compose(
        [A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=1)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )
    out = transform(image=image, bboxes=boxes, class_labels=class_labels)
    return out["image"], out["bboxes"], out["class_labels"]


# def random_scale_with_padding(image, boxes, class_labels):
#     h, w = image.shape[:2]
#     scale = np.random.normal(0.7, 0.1) if random.random() < 0.5 else np.random.normal(1.6, 0.2)
#     scale = max(0.1, min(2.5, scale))
#     transform = A.Compose(
#         [
#             A.Affine(scale=scale, fit_output=True, mode=cv2.BORDER_CONSTANT, cval=0),
#             A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, value=0),
#         ],
#         bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
#     )
#     out = transform(image=image, bboxes=boxes, class_labels=class_labels)
#     return out["image"], out["bboxes"], out["class_labels"]


def random_scale_with_padding(image, boxes, class_labels):
    h, w = image.shape[:2]

    # ── 双峰正态采样 ───────────────────────────────────────────────
    scale = np.random.normal(0.9, 0.2) if random.random() < 0.5 else np.random.normal(1.6, 0.2)
    scale = np.clip(scale, 0.7, 2.5)          # 约束可接受范围

    # ── 几何变换：缩放 + resize 回原尺寸 ─────────────────────────
    transform = A.Compose(
        [
            A.Affine(scale=scale, fit_output=False, mode=cv2.BORDER_CONSTANT, cval=0),
            A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    out = transform(image=image, bboxes=boxes, class_labels=class_labels)
    return out["image"], out["bboxes"], out["class_labels"]