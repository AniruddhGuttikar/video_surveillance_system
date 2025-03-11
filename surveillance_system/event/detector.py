import os
import cv2
import numpy as np
import datetime
import time
import subprocess
import sklearn
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

class DetectionSystem:
    def __init__(self, yolo_model="yolov8x.pt"):
        """
        Initialize the detection system
        
        Args:
            yolo_model: Path or name of YOLO model to use
        """
        # Load YOLO model
        self.detector = YOLO(yolo_model)
        
        # Initialize tracker
        self.tracks = {}
        self.next_track_id = 0
        
        # Load activity recognition model if available
        # This is just a placeholder - would need a proper activity recognition model
        self.has_activity_model = False
        
        # Initialize anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.05)
        self.anomaly_features = []
        
    def detect_objects(self, frame):
        """Detect objects in the frame using YOLO"""
        results = self.detector(frame)
        detections = []
        
        # Process detection results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def track_objects(self, frame, detections):
        """Simple tracking based on IoU overlap"""
        current_tracks = {}
        
        # If no previous tracks, assign new IDs to all detections
        if not self.tracks:
            for det in detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                det['track_id'] = track_id
                current_tracks[track_id] = det
            
            self.tracks = current_tracks
            return detections
        
        # Calculate IoU between current detections and previous tracks
        matched_dets = set()
        matched_tracks = set()
        
        for det_idx, det in enumerate(detections):
            det_bbox = det['bbox']
            
            best_iou = 0.5  # Minimum IoU threshold
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                    
                track_bbox = track['bbox']
                
                # Calculate IoU
                x1 = max(det_bbox[0], track_bbox[0])
                y1 = max(det_bbox[1], track_bbox[1])
                x2 = min(det_bbox[2], track_bbox[2])
                y2 = min(det_bbox[3], track_bbox[3])
                
                if x2 < x1 or y2 < y1:
                    continue
                
                intersection = (x2 - x1) * (y2 - y1)
                det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
                union = det_area + track_area - intersection
                
                iou = intersection / union
                
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            # If a match is found, update the track
            if best_track_id is not None:
                det['track_id'] = best_track_id
                current_tracks[best_track_id] = det
                matched_dets.add(det_idx)
                matched_tracks.add(best_track_id)
        
        # Add unmatched detections as new tracks
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                track_id = self.next_track_id
                self.next_track_id += 1
                det['track_id'] = track_id
                current_tracks[track_id] = det
        
        # Update tracks
        self.tracks = current_tracks
        
        return detections
    
    def detect_anomalies(self, tracks, frame_shape):
        """
        Detect anomalous behavior using the tracks
        
        Args:
            tracks: List of tracked objects
            frame_shape: Shape of the frame (height, width)
        
        Returns:
            List of anomalies detected
        """
        if not tracks:
            return []
        
        # Extract features for anomaly detection
        height, width = frame_shape[:2]
        
        features = []
        for track in tracks:
            # Normalize bounding box coordinates
            x1, y1, x2, y2 = track['bbox']
            bbox_center_x = (x1 + x2) / 2 / width
            bbox_center_y = (y1 + y2) / 2 / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            
            # Create feature vector
            feature = [
                bbox_center_x,
                bbox_center_y,
                bbox_width,
                bbox_height,
                track.get('confidence', 0)
            ]
            
            features.append(feature)
        
        # Store features for training
        self.anomaly_features.extend(features)
        
        # Need enough samples to train the model
        if len(self.anomaly_features) < 100:
            return []
        
        # Train the model if needed
        if len(self.anomaly_features) == 100:
            self.anomaly_detector.fit(self.anomaly_features)
        # print("number of anamoly features: ", len(self.anomaly_features))
        try:
            # Quick check to see if the model is fitted
            self.anomaly_detector.offset_  # This will raise NotFittedError if not fitted
        except (AttributeError, sklearn.exceptions.NotFittedError):
            # Model is not fitted, so fit it now
            print("Fitting anomaly detector model with", len(self.anomaly_features), "features")
            self.anomaly_detector.fit(self.anomaly_features)
        
        # Predict anomalies
        if features:  # Make sure we have features to predict on
            try:
                predictions = self.anomaly_detector.predict(features)
                
                # Find anomalous tracks
                anomalies = []
                for i, pred in enumerate(predictions):
                    if pred == -1:  # Anomaly
                        anomalies.append(tracks[i])
                
                return anomalies
            except Exception as e:
                print(f"Error predicting anomalies: {e}")
                return []
        
        return []

    def process_frame(self, frame):
        """
        Process a single frame with the detection system
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            Dictionary with detection results
        """
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Track objects
        tracked_objects = self.track_objects(frame, detections)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(tracked_objects, frame.shape)
        
        return {
            'detections': detections,
            'tracked_objects': tracked_objects,
            'anomalies': anomalies
        }
