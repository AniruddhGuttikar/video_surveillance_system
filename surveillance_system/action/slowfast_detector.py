import os
import torch
import numpy as np
import cv2
from pytorchvideo.models.slowfast import create_slowfast
from torchvision.transforms import Compose, Lambda
from torch.nn.functional import softmax
from typing import List, Dict, Any, Tuple, Optional

class SlowFastDetector:
    """
    Action recognition detector using SlowFast model.
    Detects actions in video clips and returns confidence scores.
    """

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

        self.model = create_slowfast(model_num_class=400).to(self.device)
        self.model.eval()

        if model_path:
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint.get('state_dict', checkpoint))

        # Load labels
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        else:
            self.labels = [f"action_{i}" for i in range(400)]

    def preprocess_frames(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare slow and fast pathway tensors """
        clip = torch.tensor(np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])).permute(3, 0, 1, 2).float() / 255.0
        clip = torch.nn.functional.interpolate(clip, size=(224, 224), mode='bilinear', align_corners=False)

        mean = torch.tensor([0.45, 0.45, 0.45]).view(-1, 1, 1, 1).to(self.device)
        std = torch.tensor([0.225, 0.225, 0.225]).view(-1, 1, 1, 1).to(self.device)
        clip = (clip - mean) / std

        slow_pathway = clip[:, ::self.alpha]
        fast_pathway = clip

        return slow_pathway.unsqueeze(0).to(self.device), fast_pathway.unsqueeze(0).to(self.device)

    def detect_actions(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """ Run inference on a clip and return detected actions. """
        if len(frames) < self.num_frames * self.alpha:
            frames += [frames[-1]] * ((self.num_frames * self.alpha) - len(frames))
        frames = frames[:self.num_frames * self.alpha]

        slow_pathway, fast_pathway = self.preprocess_frames(frames)

        with torch.no_grad():
            output = self.model([slow_pathway, fast_pathway])
            scores = softmax(output, dim=1).cpu().numpy()[0]

        results = [
            {"action": self.labels[idx], "confidence": float(scores[idx]), "action_id": int(idx)}
            for idx in np.argsort(-scores)[:5] if scores[idx] > self.confidence_threshold
        ]

        return results

    def detect_actions_in_video_segment(self, frames: List[np.ndarray], window_size: int = 64, stride: int = 16) -> List[Dict[str, Any]]:
        """ Sliding window action detection """
        results = []
        for start_idx in range(0, len(frames) - window_size + 1, stride):
            window_frames = frames[start_idx:start_idx + window_size]
            actions = self.detect_actions(window_frames)
            for action in actions:
                action["frame_start"] = start_idx
                action["frame_end"] = start_idx + window_size - 1
                results.append(action)
        return results
