"""
detection/anomaly_clip/model_loader.py
Loads the AnomalyCLIP model architecture with pretrained weights from the UCF-Crime checkpoint.
Provides model access and the CLIP visual encoder for downstream Re-ID feature extraction.

Assumptions:
- The checkpoint file exists at checkpoints/ucfcrime/last.ckpt (or specified path).
- The AnomalyCLIP repository is present under external/AnomalyCLIP.
- Runs on CPU or CUDA if available.
- CLIP image encoder outputs 512-dim normalized feature embeddings.
"""

import os
import sys
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import pandas as pd

# Add external/AnomalyCLIP to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
ANOMALY_CLIP_PATH = os.path.join(PROJECT_ROOT, "external", "AnomalyCLIP")
if ANOMALY_CLIP_PATH not in sys.path:
    sys.path.insert(0, ANOMALY_CLIP_PATH)

from src.models.components.anomaly_clip import AnomalyCLIP


def get_clip_transform(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])


class AnomalyCLIPWrapper:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        labels_file: Optional[str] = None,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "ucfcrime", "last.ckpt")
        self.checkpoint_path = checkpoint_path

        if labels_file is None:
            labels_file = os.path.join(ANOMALY_CLIP_PATH, "data", "ucf_labels.csv")
        self.labels_file = labels_file

        classes_df = pd.read_csv(self.labels_file)
        self.classnames = sorted(c for i, c in classes_df.values.tolist())
        self.abnormal_classes = [c for c in self.classnames if c.lower() != "normal"]
        self.normal_id = 7  # Normal in ucf_labels

        self.transform = get_clip_transform(224)
        self._load_model()

    def _load_model(self):
        config = {
            "arch": "ViT-B/16",
            "shared_context": False,
            "ctx_init": "",
            "seg_length": 16,
            "num_segments": 32,
            "select_idx_dropout_topk": 0.7,
            "select_idx_dropout_bottomk": 0.7,
            "n_ctx": 8,
            "heads": 8,
            "dim_heads": None,
            "load_from_features": True,
            "stride": 16,
            "ncrops": 1,
            "concat_features": False,
            "emb_size": 256,
            "depth": 1,
            "num_topk": 3,
            "num_bottomk": 3,
            "labels_file": self.labels_file,
            "normal_id": self.normal_id,
            "dropout_prob": 0.0,
            "temporal_module": "axial",
            "direction_module": "learned_encoder_finetune",
            "selector_module": "directions",
            "batch_norm": True,
            "feature_size": 512,
            "use_similarity_as_features": False,
        }

        self.model = AnomalyCLIP(**config)

        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = {
                k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net.")
            }
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[AnomalyCLIPWrapper] Successfully loaded weights from {self.checkpoint_path}")
        else:
            print(f"[AnomalyCLIPWrapper] Warning: Checkpoint not found at {self.checkpoint_path}, using base CLIP.")

        self.model.to(self.device)
        self.model.eval()

        # Precompute normal centroid and text features
        with torch.no_grad():
            self.text_features = self.model.get_text_features().to(self.device)
            self.ncentroid = self.text_features[self.normal_id].detach()

    @property
    def image_encoder(self) -> nn.Module:
        """Exposes the CLIP visual encoder for downstream Re-ID feature extraction."""
        return self.model.image_encoder

    def extract_image_features(self, pil_images: List[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """Extracts 512-dim CLIP visual embeddings for a list of PIL images."""
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i : i + batch_size]
                tensors = torch.stack([self.transform(img) for img in batch]).to(self.device)
                feats = self.model.image_encoder(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embeddings.append(feats.cpu())
        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        return torch.empty((0, 512))
