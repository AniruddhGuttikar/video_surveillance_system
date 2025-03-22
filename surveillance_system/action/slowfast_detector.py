import os
import torch
import numpy as np
import cv2
from pytorchvideo.models.slowfast import create_slowfast
from torchvision.transforms import Compose, Lambda
from torch.nn.functional import softmax
from typing import List, Dict, Any, Tuple, Optional

# Create our own versions of PyTorchVideo transforms to avoid the dependency issue
class UniformTemporalSubsample(torch.nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, x):
        t = x.shape[0]
        indices = torch.linspace(0, t - 1, self.num_samples).long()
        return x[indices]

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, x):
        # Move mean and std to the same device as the input tensor
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

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
            x.permute(1, 0, 2, 3),  # [t, c, h, w]
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).permute(1, 0, 2, 3)  # [c, t, h, w]
        return x

class ApplyTransformToKey:
    def __init__(self, key, transform):
        self.key = key
        self.transform = transform

    def __call__(self, data_dict):
        data_dict[self.key] = self.transform(data_dict[self.key])
        return data_dict

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
        """
        Initialize the SlowFast action detector.
        
        Args:
            model_path: Path to pretrained model weights (if None, uses default pretrained)
            device: Device to run inference on ('cuda' or 'cpu')
            labels_path: Path to labels file
            num_frames: Number of frames to sample for each clip
            alpha: Alpha parameter for SlowFast (sampling rate ratio)
            confidence_threshold: Threshold to filter detection results
        """
        self.device = device
        self.num_frames = num_frames
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        
        # Load model
        if model_path is None:
            # For newer torchvision 0.21+, we need to modify how we load the model
            self.model = create_slowfast(
                model_num_class=400,  # Kinetics-400 classes by default
            )
            # In newer PyTorchVideo, weights need to be loaded separately if available
            self.model.to(device)
        else:
            # Load from custom weights
            self.model = create_slowfast(model_num_class=400)
            # Handle different state_dict formats for compatibility
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
            except Exception as e:
                print(f"Error loading model weights: {e}")
                # Try alternative loading approach (for older format)
                checkpoint = torch.load(model_path, map_location=device)
                if "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    raise ValueError(f"Could not load model weights: {e}")
            self.model.to(device)
            
        self.model.eval()
        
        # Load action labels
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        else:
            # Use default Kinetics-400 labels or placeholder
            # Ideally, you'd download the actual labels file
            self.labels = [f"action_{i}" for i in range(400)]
        
        # Create transforms
        self.transform = self._create_transform()
    
    def _create_transform(self):
        """
        Create video transform pipeline for SlowFast input.
        """
        # Create SlowFast specific transforms
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
        """
        Extract a clip from a list of frames.
        
        Args:
            frames: List of frames (numpy arrays)
            clip_duration: Number of frames to include in clip
            
        Returns:
            Tensor containing the clip in format suitable for the model
        """
        if not frames:
            raise ValueError("No frames provided")
            
        # Ensure we have enough frames for the clip
        if len(frames) < clip_duration:
            # Pad by duplicating the last frame
            frames = frames + [frames[-1]] * (clip_duration - len(frames))
        
        # Take only clip_duration frames
        frames = frames[:clip_duration]
        
        # Convert to tensor format
        clip = torch.stack([
            torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
            for frame in frames
        ])
        
        # Move dimensions from [num_frames, channels, height, width] to [channels, num_frames, height, width]
        # which is the format expected by our transforms
        clip = clip.permute(1, 0, 2, 3)
        
        # Apply transform
        clip_dict = {"video": clip.float()}
        transformed_clip = self.transform(clip_dict)["video"]
        
        return transformed_clip
    
    def detect_actions(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect actions in a sequence of frames.
        
        Args:
            frames: List of frames (numpy arrays)
            
        Returns:
            List of detected actions with confidence scores
        """
        if not frames:
            return []
        
        # Create clip from frames
        # For SlowFast, we need a decent number of frames for temporal analysis
        min_frames_needed = self.num_frames * self.alpha
        
        if len(frames) < min_frames_needed:
            # Pad by duplicating the last frame if needed
            frames_padded = frames + [frames[-1]] * (min_frames_needed - len(frames))
        else:
            frames_padded = frames
        
        try:
            clip = self.extract_clip_from_frames(frames_padded, min_frames_needed)
            
            # Perform inference
            with torch.no_grad():
                slow_pathway, fast_pathway = clip
                slow_pathway = slow_pathway.unsqueeze(0).to(self.device)
                fast_pathway = fast_pathway.unsqueeze(0).to(self.device)
                
                output = self.model([slow_pathway, fast_pathway])
                scores = softmax(output, dim=1)
                
            # Convert to predictions
            scores_np = scores.cpu().numpy()[0]
            
            # Get top predictions above threshold
            top_indices = np.where(scores_np > self.confidence_threshold)[0]
            
            # Sort by confidence score
            top_indices = top_indices[np.argsort(-scores_np[top_indices])]
            
            # Format results
            results = []
            
            for idx in top_indices:
                results.append({
                    "action": self.labels[idx],
                    "confidence": float(scores_np[idx]),
                    "action_id": int(idx)
                })
            
            return results
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
        Process a longer video segment using sliding window approach.
        
        Args:
            frames: List of frames to process
            window_size: Number of frames in each sliding window
            stride: Number of frames to move the window each step
            
        Returns:
            Dict mapping frame indices to detected actions
        """
        if len(frames) < window_size:
            # Just process the whole segment if smaller than window
            return self.detect_actions(frames)
        
        # Use sliding window to process the video
        all_actions = []
        
        for start_idx in range(0, len(frames) - window_size + 1, stride):
            window_frames = frames[start_idx:start_idx + window_size]
            window_actions = self.detect_actions(window_frames)
            
            if window_actions:
                # Add frame range information to each action
                for action in window_actions:
                    action["frame_start"] = start_idx
                    action["frame_end"] = start_idx + window_size - 1
                    all_actions.append(action)
        
        # Merge overlapping detections (could be optimized)
        # This is a simple implementation - you may want to use a more sophisticated approach
        merged_actions = []
        action_types = set(action["action"] for action in all_actions)
        
        for action_type in action_types:
            actions_of_type = [a for a in all_actions if a["action"] == action_type]
            
            # Sort by confidence
            actions_of_type.sort(key=lambda x: x["confidence"], reverse=True)
            
            for action in actions_of_type:
                # Check if this action overlaps with any merged action
                overlap = False
                for merged in merged_actions:
                    if (merged["action"] == action_type and 
                        (merged["frame_start"] <= action["frame_end"] and 
                         merged["frame_end"] >= action["frame_start"])):
                        # Overlapping actions of same type - expand range and update confidence
                        merged["frame_start"] = min(merged["frame_start"], action["frame_start"])
                        merged["frame_end"] = max(merged["frame_end"], action["frame_end"])
                        merged["confidence"] = max(merged["confidence"], action["confidence"])
                        overlap = True
                        break
                
                if not overlap:
                    merged_actions.append(action)
        
        return merged_actions