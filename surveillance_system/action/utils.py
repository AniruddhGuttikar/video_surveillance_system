import torch
import numpy as np
import os
import cv2
from typing import List, Dict, Any, Tuple

def download_kinetics_labels():
    """
    Download Kinetics-400 labels file if not already present
    """
    import urllib.request
    import os
    
    labels_path = os.path.join(os.path.dirname(__file__), "kinetics_400_labels.txt")
    
    if not os.path.exists(labels_path):
        url = "https://raw.githubusercontent.com/Showmax/kinetics-downloader/master/resources/classes.json"
        try:
            import json
            with urllib.request.urlopen(url) as response:
                labels = json.loads(response.read())
                with open(labels_path, "w") as f:
                    for label in labels:
                        f.write(f"{label}\n")
            print(f"Downloaded Kinetics-400 labels to {labels_path}")
        except Exception as e:
            print(f"Failed to download labels: {e}")
            return None
    
    return labels_path

def extract_clips_from_frames(frames: List[np.ndarray], clip_length: int, stride: int = 8) -> List[List[np.ndarray]]:
    """
    Extract overlapping clips from a sequence of frames.
    
    Args:
        frames: List of frames
        clip_length: Number of frames per clip
        stride: Step size between clips
        
    Returns:
        List of clips (each clip is a list of frames)
    """
    if len(frames) < clip_length:
        # If we don't have enough frames, pad by duplicating the last frame
        padded_frames = frames + [frames[-1]] * (clip_length - len(frames))
        return [padded_frames]
    
    clips = []
    for i in range(0, len(frames) - clip_length + 1, stride):
        clips.append(frames[i:i + clip_length])
    
    # Add the final clip if it doesn't align with the stride
    if (len(frames) - clip_length) % stride != 0:
        clips.append(frames[-clip_length:])
    
    return clips

def merge_action_detections(action_clips: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Merge overlapping action detections from different clips.
    
    Args:
        action_clips: List of action detections from different clips
        iou_threshold: Intersection over Union threshold for merging
        
    Returns:
        List of merged actions
    """
    if not action_clips:
        return []
    
    # Sort by confidence
    sorted_actions = sorted(action_clips, key=lambda x: x["confidence"], reverse=True)
    
    merged_actions = []
    used_indices = set()
    
    for i, action in enumerate(sorted_actions):
        if i in used_indices:
            continue
        
        current_action = action.copy()
        used_indices.add(i)
        
        # Look for overlapping actions
        for j, other_action in enumerate(sorted_actions):
            if j in used_indices or j == i:
                continue
                
            if other_action["action"] != current_action["action"]:
                continue
                
            # Calculate IoU for temporal overlap
            intersection_start = max(current_action["frame_start"], other_action["frame_start"])
            intersection_end = min(current_action["frame_end"], other_action["frame_end"])
            
            if intersection_start <= intersection_end:
                intersection = intersection_end - intersection_start + 1
                union = ((current_action["frame_end"] - current_action["frame_start"] + 1) + 
                         (other_action["frame_end"] - other_action["frame_start"] + 1) - 
                         intersection)
                
                iou = intersection / union
                
                if iou >= iou_threshold:
                    # Merge the actions
                    current_action["frame_start"] = min(current_action["frame_start"], other_action["frame_start"])
                    current_action["frame_end"] = max(current_action["frame_end"], other_action["frame_end"])
                    current_action["confidence"] = max(current_action["confidence"], other_action["confidence"])
                    used_indices.add(j)
        
        merged_actions.append(current_action)
    
    return merged_actions

def map_actions_to_events(actions: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Map detected actions to existing events based on temporal overlap.
    
    Args:
        actions: List of detected actions
        events: List of events from the event processor
        
    Returns:
        Dictionary mapping event IDs to lists of actions
    """
    event_actions = {id(event): [] for event in events}
    
    for action in actions:
        action_start = action["frame_start"]
        action_end = action["frame_end"]
        
        for event in events:
            event_start = event["frame_start"]
            event_end = event["frame_end"]
            
            # Check for overlap
            if max(action_start, event_start) <= min(action_end, event_end):
                event_actions[id(event)].append(action)
    
    return event_actions