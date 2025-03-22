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
import argparse
from dotenv import load_dotenv
import os

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Import functions from surveillance_system modules
from surveillance_system.video.processor import (
    extract_video_info,
    extract_frames,
    enhance_frame,
)
from surveillance_system.event.processor import EventProcessor
from surveillance_system.event.detector import DetectionSystem

from surveillance_system.event.action_integration import ActionIntegrator

from surveillance_system.llm.surveillance_knowledge_base import SurveillanceKnowledgeBase
from surveillance_system.llm.surveillance_llm import SurveillanceLLM
from surveillance_system.llm.surveillance_llm_gemini import SurveillanceGeminiLLM


def create_streamlit_app():
    """Create a Streamlit app for the surveillance system"""
    st.title("Surveillance Video Analysis System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload surveillance video", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        # Save uploaded file
        temp_path = Path("temp_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process video button
        if st.button("Process Video"):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Extract video info and initialize event processor
            status_text.text("Extracting video information...")
            video_info = extract_video_info(str(temp_path))
            event_processor = EventProcessor(video_info)
            
            # Rest of the processing code remains the same
            # ...existing code...

def process_surveillance_video(video_path, output_dir="output", api_key=None):
    """
    Process a surveillance video through the complete pipeline with action recognition
    
    Args:
        video_path: Path to the video file
        output_dir: Directory for output files
        api_key: API key for LLM and vector database
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    surveillance_knowledge_base = SurveillanceKnowledgeBase(api_key=PINECONE_API_KEY)
    surveillance_llm_gemini = SurveillanceGeminiLLM(api_key=GEMINI_API_KEY)
    
    print("Extracting video information...")

    # Initialize required classes
    video_info = extract_video_info(video_path)
    print("video info: ", video_info)

    event_processor = EventProcessor(video_info)
    detection_system = DetectionSystem()
    
    # Extract frames
    print("Extracting frames...")
    frames_dir = os.path.join(output_dir, "frames")
    frames = extract_frames(video_path, frames_dir)
    
    print("Processing frames with object detection...")
    detection_results = []
    
    # Process frames with object detection
    for i in range(0, len(frames), 15):
        frame = frames[i]
        frame_img = cv2.imread(frame["path"])
        if frame_img is None:
            print(f"Warning: Could not read image from {frame}, skipping...")
            continue

        enhanced_frame = enhance_frame(frame_img, False, False)
        results = detection_system.process_frame(enhanced_frame)
        detection_results.append(results)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(frames)} frames")
    
    # Initialize action integrator
    action_integrator = ActionIntegrator()
    
    print("Detecting actions in video...")
    # Process entire video with action recognition
    actions = action_integrator.process_frames(frames)
    
    print(f"Detected {len(actions)} actions")
    for action in actions[:5]:  # Print sample actions
        print(f"Action: {action['action']}, Confidence: {action['confidence']:.2f}, Frames: {action['frame_start']}-{action['frame_end']}")
    
    # Create regular events from detection results
    print("Processing events...")
    events = []
    for i, results in enumerate(detection_results):
        frame_idx = i * 15  # Since we're processing every 15th frame
        current_events = event_processor.process_detection_results(frame_idx, results)
        events.extend(current_events)
    
    # Enhance events with action information
    enhanced_events = action_integrator.enhance_events_with_actions(events, actions)
    
    print(f"Total events identified: {len(enhanced_events)}")
    
    # Generate action statistics
    action_counts = {}
    for event in enhanced_events:
        if "actions" in event:
            for action_info in event["actions"]:
                action = action_info["action"]
                action_counts[action] = action_counts.get(action, 0) + 1
    
    action_stats = {"action_counts": action_counts}
    
    # Generate event descriptions with action information
    print("Generating event descriptions...")
    event_descriptions_map = surveillance_llm_gemini.generate_batch_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    processed_data = process_surveillance_video(args.video_path)
    print(processed_data)
