import os
import cv2
import numpy as np
import datetime
import time
import subprocess
import torch
from ultralytics import YOLO
from PIL import Image
import json
import pandas as pd
from sklearn.ensemble import IsolationForest
import anthropic
from typing import List, Dict, Any, Tuple
import pinecone
from sentence_transformers import SentenceTransformer
import streamlit as st
from pathlib import Path
import re

from surveillance_system.event.action_integration import ActionIntegrator

# ========== 4. EVENT PROCESSING ==========

class EventProcessor:
    def __init__(self, video_info):
        """
        Initialize the event processor
        
        Args:
            video_info: Dictionary with video metadata
        """
        self.video_info = video_info
        self.events = []
        self.last_event_time = {}  # Track last event time by type
        self.min_event_interval = 3.0  # Minimum seconds between events of same type
        
    def frame_to_timestamp(self, frame_idx):
        """Convert frame index to absolute timestamp"""
        if not self.video_info or 'fps' not in self.video_info:
            return None
        # print(f"frame_idx: {frame_idx}, video_info: {self.video_info['fps']}")
        seconds = frame_idx / self.video_info['fps']
        
        # If we have video start time, calculate absolute timestamp
        if 'start_time' in self.video_info and self.video_info['start_time']:
            return self.video_info['start_time'] + datetime.timedelta(seconds=seconds)
        
        # Otherwise return relative timestamp
        return datetime.timedelta(seconds=seconds)
    
    def process_detection_results(self, frame_idx, results):
        """
        Process detection results and generate events
        
        Args:
            frame_idx: Index of the current frame
            results: Detection results from detection system
        
        Returns:
            List of events generated from this frame
        """
        current_events = []
        timestamp = self.frame_to_timestamp(frame_idx)
        
        # Process regular detections
        for detection in results['detections']:
            event_type = f"detected_{detection['class_name']}"
            
            # Check if we should create a new event based on time interval
            last_time = self.last_event_time.get(event_type)
            current_time = frame_idx / self.video_info['fps']
            
            if last_time is None or (current_time - last_time) >= self.min_event_interval:
                # Create new event
                event = {
                    'type': event_type,
                    'subtype': 'detection',
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'confidence': detection['confidence'],
                    'bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'track_id': detection.get('track_id')
                }
                
                current_events.append(event)
                self.last_event_time[event_type] = current_time
        
        # Process anomalies (considered high priority)
        for anomaly in results['anomalies']:
            event = {
                'type': 'anomaly',
                'subtype': f"anomalous_{anomaly['class_name']}",
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'confidence': anomaly['confidence'],
                'bbox': anomaly['bbox'],
                'class_name': anomaly['class_name'],
                'track_id': anomaly.get('track_id'),
                'priority': 'high'
            }
            
            current_events.append(event)
        
        # Add events to the global list
        self.events.extend(current_events)
        
        return current_events
    
    def get_events_in_timerange(self, start_time, end_time):
        """
        Get events within a specific time range
        
        Args:
            start_time: Start time (datetime or timedelta)
            end_time: End time (datetime or timedelta)
        
        Returns:
            List of events within the time range
        """
        result = []
        
        for event in self.events:
            event_time = event['timestamp']
            if start_time <= event_time <= end_time:
                result.append(event)
        
        return result
    
    def get_events_by_type(self, event_type):
        """Get events of a specific type"""
        return [e for e in self.events if e['type'] == event_type]
    
    def get_events_by_object(self, class_name):
        """Get events involving a specific object class"""
        return [e for e in self.events if e.get('class_name') == class_name]
    
    def export_events(self, events, output_file="events.json"):
        """Export events to a JSON file"""
        # Convert timestamps to strings for JSON serialization
        serializable_events = []
        events = events if events else self.events
        
        for event in events:
            serializable_events.append(event)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_events, f, indent=2)
            
    def process_with_action_recognition(
        self, 
        frames: List[Dict[str, str]], 
        detection_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process frames with both object detection and action recognition.
        
        Args:
            frames: List of frame dictionaries
            detection_results: Results from detection system
            
        Returns:
            Enhanced events with action information
        """
        # First process with standard event detection
        events = []
        for i, result in enumerate(detection_results):
            current_events = self.process_detection_results(i, result)
            events.extend(current_events)
        
        # Initialize action integrator
        action_integrator = ActionIntegrator()
        
        # Detect actions in frames
        actions = action_integrator.process_frames(frames)
        
        # Enhance events with action information
        enhanced_events = action_integrator.enhance_events_with_actions(events, actions)
        
        return enhanced_events
    
    def export_events_with_actions(
        self, 
        events: List[Dict[str, Any]], 
        event_descriptions: List[str], 
        output_path: str
    ):
        """
        Export events with action information to a JSON file.
        
        Args:
            events: List of events with action information
            event_descriptions: List of event descriptions
            output_path: Path to save the JSON file
        """
        # Create exportable events with descriptions
        exportable_events = []
        
        for i, event in enumerate(events):
            event_data = {
                "id": i,
                "type": event["type"],
                "start_time": self.frame_to_timestamp(event["frame_start"]),
                "end_time": self.frame_to_timestamp(event["frame_end"]),
                "frame_start": event["frame_start"],
                "frame_end": event["frame_end"],
                "objects": event["objects"],
                "description": event_descriptions[i] if i < len(event_descriptions) else "",
            }
            
            # Add action data if available
            if "actions" in event:
                event_data["actions"] = event["actions"]
                event_data["primary_action"] = event["primary_action"]
                event_data["action_confidence"] = event["action_confidence"]
            
            exportable_events.append(event_data)
        
        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(exportable_events, f, indent=2)
