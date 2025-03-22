import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Optional

from surveillance_system.action.slowfast_detector import SlowFastDetector
from surveillance_system.action.utils import (
    merge_action_detections,
    map_actions_to_events,
    download_kinetics_labels
)

class ActionIntegrator:
    """
    Integrates action recognition with event processing pipeline.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        confidence_threshold: float = 0.4,
        clip_length: int = 64,
        clip_stride: int = 16
    ):
        """
        Initialize the action integrator.
        
        Args:
            model_path: Path to custom model (None uses pretrained)
            device: Device for inference
            confidence_threshold: Threshold for action detection
            clip_length: Length of clips to process
            clip_stride: Stride between clips
        """
        # Download labels if needed
        labels_path = download_kinetics_labels()
        
        # Initialize SlowFast detector
        self.detector = SlowFastDetector(
            model_path=model_path,
            device=device,
            labels_path=labels_path,
            confidence_threshold=confidence_threshold
        )
        
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        
    def process_frames(
        self, 
        frames: List[Dict[str, str]],
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Process frames and detect actions.
        
        Args:
            frames: List of frame dictionaries from extract_frames
            batch_size: Number of frames to process at once
            
        Returns:
            List of detected actions
        """
        all_actions = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            # Load frames into memory
            loaded_frames = []
            for frame_dict in batch_frames:
                frame = cv2.imread(frame_dict["path"])
                if frame is not None:
                    loaded_frames.append(frame)
            
            # Detect actions in this batch
            batch_actions = self.detector.detect_actions_in_video_segment(
                loaded_frames,
                window_size=min(self.clip_length, len(loaded_frames)),
                stride=self.clip_stride
            )
            
            # Adjust frame indices to global indices
            for action in batch_actions:
                action["frame_start"] += i
                action["frame_end"] += i
                
            all_actions.extend(batch_actions)
        
        # Merge overlapping detections
        from surveillance_system.action.utils import merge_action_detections
        merged_actions = merge_action_detections(all_actions)
        
        return merged_actions
    
    def enhance_events_with_actions(
        self, 
        events: List[Dict[str, Any]], 
        actions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhance event data with detected actions.
        
        Args:
            events: List of events from EventProcessor
            actions: List of detected actions
            
        Returns:
            Events with added action information
        """
        # Map actions to events
        event_actions = map_actions_to_events(actions, events)
        
        # Enhance each event with action information
        enhanced_events = []
        for event in events:
            event_id = id(event)
            event_copy = event.copy()
            
            if event_id in event_actions and event_actions[event_id]:
                # Add actions to event
                actions_list = event_actions[event_id]
                
                # Sort by confidence
                actions_list.sort(key=lambda x: x["confidence"], reverse=True)
                
                # Add to event
                event_copy["actions"] = actions_list
                
                # Set primary action (highest confidence)
                event_copy["primary_action"] = actions_list[0]["action"]
                event_copy["action_confidence"] = actions_list[0]["confidence"]
            else:
                # No actions detected for this event
                event_copy["actions"] = []
                event_copy["primary_action"] = "unknown"
                event_copy["action_confidence"] = 0.0
                
            enhanced_events.append(event_copy)
            
        return enhanced_events