# final_inference.py

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
import timm
import torch.nn as nn
# -------------------------------
# 1) CONFIGURATION / PATHS
# -------------------------------

# Path to your custom YOLOv8n weights (ped-button detector)
YOLO_WEIGHTS = 'TwoStageYOLOTrain/train5v8n/weights/best.pt'

# Path to your Left/Right classifier .pth
CLS_STATE_DICT = 'results/Dinov2_w_pretrain/dinov2_small_256_.5_final.pth'  

# Device: use GPU if available, else CPU
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ------------------------------------------------
# 2) LOAD YOLOv8n (ped-button detector)
# ------------------------------------------------

print(f"Loading YOLOv8 model from: {YOLO_WEIGHTS}  →  {DEVICE}")
# By default, YOLO(...) will load onto CPU unless you pass device=...
yolo_model = YOLO(YOLO_WEIGHTS)  # automatically chooses best backend
# You can optionally set a confidence threshold here or pass it at inference time
yolo_model.conf = 0.25  # keep detections with ≥25% confidence

# ---------------------------------------------------
# 3) RECONSTRUCT & LOAD THE Dinov2BinaryClassifier
# ---------------------------------------------------
from torch import nn
from transformers import AutoImageProcessor, Dinov2Model

print(f"Loading DINOv2-based classifier from: {CLS_STATE_DICT}")

class Dinov2BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # Output: logit
        )

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [B, 384]
        return self.classifier(cls_token)

classifier = Dinov2BinaryClassifier()
state_dict = torch.load(CLS_STATE_DICT, map_location=DEVICE)
classifier.load_state_dict(state_dict)
classifier.to(DEVICE).eval()

print("→ DINOv2 classifier is ready.")




# ----------------------------------------------------
# 4) TRANSFORMS FOR THE Dinov2BinaryClassifier
# ----------------------------------------------------
from torchvision.transforms import ToPILImage
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

def dinov2_transform(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = ToPILImage()(image_rgb)
    processed = image_processor(images=pil_image, return_tensors="pt")
    return processed["pixel_values"].to(DEVICE)  # [1, 3, H, W]



# ---------------------------------------------------
# 5) RUN REAL-TIME “DETECT → CROP → CLASSIFY” LOOP
# ---------------------------------------------------

def detect_and_classify(frame_bgr: np.ndarray) -> np.ndarray:
    """
    1) Resize input BGR frame to 720×720
    2) Run YOLOv8 → get bounding boxes
    3) For each box: crop, transform → classifier → “Left”/“Right”
    4) Draw boxes + labels on the 720×720 frame and return it.
    """
    # 5a) Resize to 720×720 (YOLOv8 will accept any size, but we want consistent scaling)
    img720 = cv2.resize(frame_bgr, (720, 720))

    # 5b) Run YOLOv8 inference on the 720×720 BGR image
    #    (Ultralytics automatically does BGR→RGB, normalization, batch dim, etc.)
    results = yolo_model(
    img720,
    device=DEVICE,   # "cuda:0" or "cpu"
    imgsz=720,       # same size used during training
    conf=0.25        # confirm threshold
    )[0]

    # 5c) Convert results.boxes to numpy for iteration
    # boxes.xyxy: tensor of shape [N, 4], boxes.conf: [N], boxes.cls: [N]
    if results.boxes is not None and len(results.boxes) > 0:
        # Draw onto a copy of img720 (so we can crop reliably)
        annotated = img720.copy()

        # Extract numpy arrays
        xyxy_tensor = results.boxes.xyxy.cpu().numpy()   # shape [N,4]
        confs = results.boxes.conf.cpu().numpy()         # shape [N]
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)  # shape [N]

        for i, (box, conf, cls_id) in enumerate(zip(xyxy_tensor, confs, cls_ids)):
            # If YOLOv8 was trained with multiple classes, you can check cls_id here.
            # For example, if “ped button” = 0, then skip any other cls_id ≠ 0.
            # if cls_id != 0:
            #     continue

            x1, y1, x2, y2 = map(int, box)

            # 5d) Crop the detected region from the 720×720 frame
            crop = annotated[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 5e) Preprocess crop → classifier input
            inp_tensor = dinov2_transform(crop)

            # 5f) Run classifier: get logit → “Left”/“Right”
            with torch.no_grad():
                logit = classifier(inp_tensor)  # [1, 1]
                pred_idx = (torch.sigmoid(logit) > 0.5).item()
                label = "Right" if pred_idx else "Left"


            # 5g) Draw bbox + label on annotated image
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return annotated

    else:
        # If no boxes, just return the plain resized frame
        return img720


def main():
    cap = cv2.VideoCapture(1)  # 0 = default webcam
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        # Run the two-stage detector→classifier
        output_img = detect_and_classify(frame)

        # Display the 720×720 result
        cv2.imshow("YOLOv8n → Crop → Left/Right Classifier", output_img)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
