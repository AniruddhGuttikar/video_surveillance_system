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

# ========== 1. VIDEO INPUT PROCESSING ==========

def extract_video_info(video_path):
    """Extract metadata from video file using FFmpeg"""
    cmd = [
        'ffmpeg', 
        '-i', video_path, 
        '-hide_banner'
    ]
    
    # Run FFmpeg command and capture output
    process = subprocess.Popen(
        cmd, 
        stderr=subprocess.PIPE, 
        stdout=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    
    # Parse the output
    output = stderr.decode('utf-8')
    
    # Extract video information
    info = {
        'path': video_path,
        'filename': os.path.basename(video_path),
        'fps': None,
        'duration': None,
        'resolution': None,
        'start_time': None
    }
    
    # Extract fps
    fps_match = re.search(r'(\d+\.?\d*) fps', output)
    if fps_match:
        info['fps'] = float(fps_match.group(1))
    
    # Extract duration
    duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', output)
    if duration_match:
        hours, minutes, seconds = duration_match.groups()
        info['duration'] = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    
    # Extract resolution
    resolution_match = re.search(r'(\d+x\d+)', output)
    if resolution_match:
        info['resolution'] = resolution_match.group(1)
    
    # Try to extract creation time if available
    creation_time_match = re.search(r'creation_time\s*:\s*([\d\-]+\s+[\d:]+)', output)
    if creation_time_match:
        time_str = creation_time_match.group(1)
        try:
            # Parse the timestamp
            info['start_time'] = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass
    
    return info

# ========== 2. FRAME EXTRACTION & ANALYSIS ==========

def extract_frames(video_path, output_dir="frames", sample_rate=1.0, detect_scene_change=True):
    """
    Extract frames from video with adaptive sampling based on scene changes
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        sample_rate: Base frame sampling rate (1.0 = every frame)
        detect_scene_change: Whether to increase sampling rate during scene changes
    
    Returns:
        List of extracted frame paths and their timestamps
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise Exception(f"Could not open video file {video_path}")
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate base frame interval
    base_interval = int(1 / sample_rate)
    
    # Initialize variables
    prev_frame = None
    frame_info = []
    frame_idx = 0
    
    while True:
        # Read the next frame
        ret, frame = video.read()
        if not ret:
            break
        
        should_save = False
        
        # Base sampling
        if frame_idx % base_interval == 0:
            should_save = True
        
        # Scene change detection
        if detect_scene_change and prev_frame is not None:
            # Convert frames to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate difference between frames
            frame_diff = cv2.absdiff(gray_frame, gray_prev_frame)
            mean_diff = np.mean(frame_diff)
            
            # If significant change, save the frame
            if mean_diff > 10.0:  # Threshold can be adjusted
                should_save = True
        
        if should_save:
            # Calculate timestamp
            timestamp = frame_idx / fps
            formatted_time = str(datetime.timedelta(seconds=timestamp))
            
            # Save the frame
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Store frame info
            frame_info.append({
                'frame_idx': frame_idx,
                'path': frame_path,
                'timestamp': timestamp,
                'formatted_time': formatted_time
            })
        
        # Update previous frame
        prev_frame = frame.copy()
        frame_idx += 1
    
    # Release the video capture object
    video.release()
    
    return frame_info

def enhance_frame(frame, denoise=True, contrast=True):
    """
    Enhance image quality for better detection
    
    Args:
        frame: Input frame (numpy array)
        denoise: Whether to apply denoising
        contrast: Whether to improve contrast
    
    Returns:
        Enhanced frame
    """
    enhanced = frame.copy()
    
    # Apply denoising if requested
    if denoise:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    # Improve contrast if requested
    if contrast:
        # Convert to LAB color space
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        
        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L channel with A and B channels
        merged = cv2.merge((cl, a, b))
        
        # Convert back to BGR color space
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return enhanced