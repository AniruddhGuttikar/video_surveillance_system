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
            
    def detect_actions_in_video_segment(
        self, 
        frames: List[np.ndarray],
        window_size: int = 64,
        stride: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Detect actions in a segment of video frames using sliding windows.
        
        Args:
            frames: List of video frames as numpy arrays
            window_size: Size of sliding window (number of frames)
            stride: Step size for sliding window
            
        Returns:
            List of action detections with frame indices
        """
        if not frames or len(frames) < self.num_frames:
            return []
            
        # Ensure window size is at least the required number of frames
        window_size = max(window_size, self.num_frames * self.alpha)
        
        # Adjust window size if it's larger than available frames
        window_size = min(window_size, len(frames))
        
        all_detections = []
        
        # Slide a window through the frames
        for start_idx in range(0, len(frames) - window_size + 1, stride):
            end_idx = start_idx + window_size
            
            # Get window of frames
            window_frames = frames[start_idx:end_idx]
            
            # Detect actions in this window
            actions = self.detect_actions(window_frames)
            
            # Add frame indices to detections
            for action in actions:
                detection = action.copy()
                detection["frame_start"] = start_idx
                detection["frame_end"] = end_idx - 1
                all_detections.append(detection)
                
        return all_detections
    
    def detect_actions_with_tracking(
        self, 
        frames: List[np.ndarray], 
        person_detections: List[Dict[str, Any]],
        crop_padding: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Detect actions for specific tracked people in the video.
        
        Args:
            frames: List of video frames
            person_detections: List of person detection dicts with bbox info
            crop_padding: Padding around bounding boxes
            
        Returns:
            List of actions with person IDs
        """
        if not frames or not person_detections:
            return []
            
        # Group detections by person ID
        person_tracks = {}
        for detection in person_detections:
            person_id = detection.get("person_id", detection.get("id"))
            if person_id not in person_tracks:
                person_tracks[person_id] = []
            person_tracks[person_id].append(detection)
            
        all_actions = []
        
        # Process each person track
        for person_id, detections in person_tracks.items():
            if len(detections) < self.num_frames * self.alpha:
                continue
                
            # Extract cropped frames for this person
            person_frames = []
            frame_indices = []
            
            for detection in detections:
                frame_idx = detection.get("frame_idx")
                if frame_idx is None or frame_idx >= len(frames):
                    continue
                    
                frame = frames[frame_idx]
                x1, y1, x2, y2 = self._get_bbox_coords(detection, frame.shape, crop_padding)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                cropped_frame = frame[y1:y2, x1:x2]
                if cropped_frame.size == 0:
                    continue
                    
                # Resize to ensure consistent dimensions
                cropped_frame = cv2.resize(cropped_frame, (224, 224))
                
                person_frames.append(cropped_frame)
                frame_indices.append(frame_idx)
                
            # Skip if not enough frames
            if len(person_frames) < self.num_frames * self.alpha:
                continue
                
            # Detect actions for this person
            actions = self.detect_actions(person_frames)
            
            # Add person ID and frame index information
            for action in actions:
                action["person_id"] = person_id
                action["frame_start"] = frame_indices[0]
                action["frame_end"] = frame_indices[-1]
                all_actions.append(action)
                
        return all_actions
        
    def _get_bbox_coords(
        self, 
        detection: Dict[str, Any], 
        frame_shape: Tuple[int, int, int],
        padding: int = 10
    ) -> Tuple[int, int, int, int]:
        """
        Extract bbox coordinates from detection with padding.
        
        Args:
            detection: Detection dictionary
            frame_shape: Shape of the frame
            padding: Padding to add around bbox
            
        Returns:
            Tuple of (x1, y1, x2, y2) coordinates
        """
        height, width = frame_shape[:2]
        
        # Handle different bbox formats
        if "bbox" in detection:
            bbox = detection["bbox"]
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            elif len(bbox) == 4 and isinstance(bbox[2], int) and isinstance(bbox[3], int):
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
        elif all(k in detection for k in ["x1", "y1", "x2", "y2"]):
            x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        elif all(k in detection for k in ["left", "top", "right", "bottom"]):
            x1, y1, x2, y2 = detection["left"], detection["top"], detection["right"], detection["bottom"]
        else:
            # Default to full frame if no bbox found
            return 0, 0, width, height
            
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        return int(x1), int(y1), int(x2), int(y2)