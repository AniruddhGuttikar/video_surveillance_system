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
    
    def export_events(self, output_file="events.json"):
        """Export events to a JSON file"""
        # Convert timestamps to strings for JSON serialization
        serializable_events = []
        
        for event in self.events:
            event_copy = event.copy()
            event_copy['timestamp'] = str(event['timestamp'])
            serializable_events.append(event_copy)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_events, f, indent=2)
