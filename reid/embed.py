"""
reid/embed.py
Subject detector and CLIP embedding extractor.

Assumptions:
- Person/vehicle detection using YOLOv8 (ultralytics).
- Reuses the EXACT same CLIP image encoder from AnomalyCLIP (does not load a second embedding model).
- Embeddings are 512-dim normalized vectors matching Qdrant cosine space.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from detection.anomaly_clip.model_loader import AnomalyCLIPWrapper, get_clip_transform


class SubjectEmbedder:
    def __init__(
        self,
        anomaly_clip_wrapper: Optional[AnomalyCLIPWrapper] = None,
        yolo_model_name: str = "yolov8n.pt",
    ):
        # 1. Reuse existing CLIP visual backbone from AnomalyCLIP
        self.clip_wrapper = anomaly_clip_wrapper or AnomalyCLIPWrapper()
        self.image_encoder = self.clip_wrapper.image_encoder
        self.transform = self.clip_wrapper.transform
        self.device = self.clip_wrapper.device

        # 2. YOLOv8 detector for person / vehicle
        print(f"[SubjectEmbedder] Loading YOLOv8 model: {yolo_model_name}")
        self.yolo = YOLO(yolo_model_name)
        # Class IDs in COCO: 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [0, 2, 3, 5, 7]

    def detect_and_crop(
        self, frame_bgr: np.ndarray, conf_threshold: float = 0.35
    ) -> List[Tuple[Image.Image, str, List[int]]]:
        """
        Runs YOLOv8 and crops detected persons and vehicles.
        Returns: List of (PIL Image crop, class_name, bbox [x1, y1, x2, y2])
        """
        results = self.yolo.predict(frame_bgr, conf=conf_threshold, classes=self.target_classes, verbose=False)
        crops = []
        if not results:
            return crops

        boxes = results[0].boxes
        if boxes is None:
            return crops

        H, W = frame_bgr.shape[:2]
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.yolo.names[cls_id]
            xyxy = [int(v.item()) for v in box.xyxy[0]]
            x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(W, xyxy[2]), min(H, xyxy[3])

            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue  # Filter tiny crops

            crop_bgr = frame_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            crops.append((pil_crop, cls_name, [x1, y1, x2, y2]))

        return crops

    def embed_crops(self, crops: List[Image.Image]) -> np.ndarray:
        """
        Extracts normalized 512-dim CLIP visual embeddings using the shared AnomalyCLIP encoder.
        """
        if not crops:
            return np.empty((0, 512), dtype=np.float32)

        tensors = torch.stack([self.transform(c) for c in crops]).to(self.device)
        with torch.no_grad():
            feats = self.image_encoder(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype(np.float32)
