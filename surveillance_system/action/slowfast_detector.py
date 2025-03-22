import os
import torch
import numpy as np
import cv2
from pytorchvideo.models.slowfast import create_slowfast
from torchvision.transforms import Compose, Lambda
from torch.nn.functional import softmax
from typing import List, Dict, Any, Tuple, Optional

# Create custom transforms to avoid dependency issues
class UniformTemporalSubsample(torch.nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, x):
        t = x.shape[1]
        indices = torch.linspace(0, t - 1, self.num_samples).long().to(x.device)
        return x[:, indices]

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(torch.float32)
        self.std = torch.tensor(std).view(-1, 1, 1).to(torch.float32)

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

class ShortSideScale(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        c, t, h, w = x.shape
        if h < w:
            new_h = self.size
            new_w = int(w * (new_h / h))
        else:
            new_w = self.size
            new_h = int(h * (new_w / w))

        x = torch.nn.functional.interpolate(
            x.permute(1, 0, 2, 3),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).permute(1, 0, 2, 3)
        return x

class ApplyTransformToKey:
    def __init__(self, key, transform):
        self.key = key
        self.transform = transform

    def __call__(self, data_dict):
        data_dict[self.key] = self.transform(data_dict[self.key])
        return data_dict

class SlowFastDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        labels_path: Optional[str] = None,
        num_frames: int = 32,
        alpha: int = 4,
        confidence_threshold: float = 0.4
    ):
        self.device = device
        self.num_frames = num_frames
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold

        if model_path is None:
            self.model = create_slowfast(model_num_class=400)
        else:
            self.model = create_slowfast(model_num_class=400)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint.get("state_dict", checkpoint))

        self.model.to(self.device)
        self.model.eval()

        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        else:
            self.labels = [f"action_{i}" for i in range(400)]

        self.transform = self._create_transform()

    def _create_transform(self):
        slow_pathway_transform = Compose([
            UniformTemporalSubsample(self.num_frames),
            Lambda(lambda x: x / 255.0),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ])

        fast_pathway_transform = Compose([
            UniformTemporalSubsample(self.num_frames * self.alpha),
            Lambda(lambda x: x / 255.0),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ])

        return ApplyTransformToKey(
            key="video",
            transform=Compose([
                ShortSideScale(224),
                Lambda(lambda x: (
                    slow_pathway_transform(x),
                    fast_pathway_transform(x),
                )),
            ]),
        )

    def extract_clip_from_frames(self, frames: List[np.ndarray], clip_duration: int) -> torch.Tensor:
        frames = frames[:clip_duration]
        clip = torch.stack([
            torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).to(self.device)
            for frame in frames
        ])

        clip = clip.permute(1, 0, 2, 3).float()
        clip_dict = {"video": clip}
        return self.transform(clip_dict)["video"]

    def detect_actions(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        if not frames:
            return []

        try:
            clip = self.extract_clip_from_frames(frames, self.num_frames * self.alpha)
            with torch.no_grad():
                slow_pathway, fast_pathway = clip
                slow_pathway = slow_pathway.unsqueeze(0).to(self.device)
                fast_pathway = fast_pathway.unsqueeze(0).to(self.device)

                output = self.model([slow_pathway, fast_pathway])
                scores = softmax(output, dim=1)

            scores_np = scores.cpu().numpy()[0]
            top_indices = np.where(scores_np > self.confidence_threshold)[0]
            top_indices = top_indices[np.argsort(-scores_np[top_indices])]

            return [{
                "action": self.labels[idx],
                "confidence": float(scores_np[idx]),
                "action_id": int(idx)
            } for idx in top_indices]

        except Exception as e:
            print(f"Error in detect_actions: {e}")
            return []
