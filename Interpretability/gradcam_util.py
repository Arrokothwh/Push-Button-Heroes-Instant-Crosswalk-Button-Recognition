import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import cv2
import os
import math
from PIL import Image
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms as transforms



def create_image_grid(images, images_per_row=4, padding=10, bg_color=(255, 255, 255)):
    """
    stiching pics
    """
    if len(images) == 0:
        return None

    img_width, img_height = images[0].size
    n_images = len(images)
    n_rows = math.ceil(n_images / images_per_row)

    grid_width = images_per_row * img_width + (images_per_row - 1) * padding
    grid_height = n_rows * img_height + (n_rows - 1) * padding

    grid_img = Image.new("RGB", (grid_width, grid_height), bg_color)

    for idx, img in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        x = col * (img_width + padding)
        y = row * (img_height + padding)
        grid_img.paste(img, (x, y))

    return grid_img

def get_target_layers_and_transform(model_name, model):
    if model_name == "dinov2":
        target_layers = [model.backbone.encoder.layer[-1].norm1]
        def reshape_transform(x):
            x = x[:, 1:, :]
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            return x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    elif model_name == "tinyvit":
        target_layers = [model.backbone.stages[2].blocks[-1].local_conv.conv]
        reshape_transform = None
    elif model_name == "resnet":
        target_layers = [model.backbone.layer4[-1].conv2]
        reshape_transform = None
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return target_layers, reshape_transform

def batch_grad_cam(model_name, model, batch_dir, exp = 0.8):
    all_tensors = []
    file_list = []

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ])

    # === Load images and convert to tensors ===
    for filename in sorted(os.listdir(batch_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(batch_dir, filename)
            img = Image.open(image_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            all_tensors.append(tensor)
            file_list.append(filename)

    if not all_tensors:
        raise RuntimeError(f"No images found in {batch_dir}.")

    # === Get target layers and reshape_transform ===
    target_layers, reshape_transform = get_target_layers_and_transform(model_name, model)

    results = []
    with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        for tensor, filename in zip(all_tensors, file_list):
            input_tensor = tensor  # shape (1, 3, H, W)

            # GradCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0]
            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
            grayscale_cam = grayscale_cam ** exp

            # 原图和预测
            img_path = os.path.join(batch_dir, filename)
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_bgr = img_np[..., ::-1]

            cam_image = show_cam_on_image(img_bgr, grayscale_cam, use_rgb=True, image_weight=0.4)

            logit = model(input_tensor)
            score = torch.sigmoid(logit).item()
            label = "left" if score < 0.5 else "right"
            display_score = 1.0 - score if label == "left" else score
            text = f"{label} ({display_score:.2f})"

            cam_with_text = cam_image.copy()
            cv2.putText(cam_with_text, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

            img_uint8 = np.uint8(255 * img_np)
            combined = np.hstack((img_uint8, cam_with_text))
            results.append(Image.fromarray(combined))

    return create_image_grid(results, images_per_row=4)

def batch_hires_cam(model_name, model, batch_dir, exp = 0.8):
    all_tensors = []
    file_list = []

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ])

    # === Load images and convert to tensors ===
    for filename in sorted(os.listdir(batch_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(batch_dir, filename)
            img = Image.open(image_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            all_tensors.append(tensor)
            file_list.append(filename)

    if not all_tensors:
        raise RuntimeError(f"No images found in {batch_dir}.")

    # === Get target layers and reshape_transform ===
    target_layers, reshape_transform = get_target_layers_and_transform(model_name, model)

    results = []
    with HiResCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        for tensor, filename in zip(all_tensors, file_list):
            input_tensor = tensor  # shape (1, 3, H, W)

            # GradCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0]
            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
            grayscale_cam = grayscale_cam ** exp

            # 原图和预测
            img_path = os.path.join(batch_dir, filename)
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_bgr = img_np[..., ::-1]

            cam_image = show_cam_on_image(img_bgr, grayscale_cam, use_rgb=True, image_weight=0.4)

            logit = model(input_tensor)
            score = torch.sigmoid(logit).item()
            label = "left" if score < 0.5 else "right"
            display_score = 1.0 - score if label == "left" else score
            text = f"{label} ({display_score:.2f})"

            cam_with_text = cam_image.copy()
            cv2.putText(cam_with_text, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

            img_uint8 = np.uint8(255 * img_np)
            combined = np.hstack((img_uint8, cam_with_text))
            results.append(Image.fromarray(combined))

    return create_image_grid(results, images_per_row=4)
