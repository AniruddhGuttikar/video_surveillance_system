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

# Import functions from surveillance_system modules
from surveillance_system.video.processor import (
    extract_video_info,
    extract_frames,
    enhance_frame,
)
from surveillance_system.event.processor import EventProcessor
from surveillance_system.event.detector import DetectionSystem

from surveillance_system.llm.surveillance_knowledge_base import SurveillanceKnowledgeBase
from surveillance_system.llm.surveillance_llm import SurveillanceLLM

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
    Process a surveillance video through the complete pipeline
    
    Args:
        video_path: Path to the video file
        output_dir: Directory for output files
        api_key: API key for LLM and vector database
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Extracting video information...")

    # Initialize required classes
    video_info = extract_video_info(video_path)

    event_processor = EventProcessor(video_info)
    detection_system = DetectionSystem()
    
    # Rest of the pipeline remains the same but using initialized classes
    print("Extracting frames...")
    frames_dir = os.path.join(output_dir, "frames")
    frame_paths = extract_frames(video_path, frames_dir)
    
    print("Processing frames...")
    events = []
    for i, frame_path in enumerate(frame_paths):
        enhanced_frame = enhance_frame(frame_path)
        timestamp = event_processor.frame_to_timestamp(i, video_info)
        
        # Use detection_system instance
        detections = detection_system.detect_objects(enhanced_frame)
        tracking_results = detection_system.track_objects(detections, i)
        
        # Use event_processor instance
        processed_data = event_processor.process_detection_results(tracking_results, timestamp)
        anomalies = event_processor.detect_anomalies(processed_data, tracking_results)
        
        for anomaly in anomalies:
            event_processor.add_event(anomaly)
            events.append(anomaly)
            
        if i % 100 == 0:
            print(f"Processed {i}/{len(frame_paths)} frames")
    
    # Step 4: Generate event descriptions and summaries
    surveillance_knowledge_base = SurveillanceKnowledgeBase()
    surveillance_llm = SurveillanceLLM()
    print("Generating event descriptions...")
    event_descriptions = []
    for event in events:
        description = surveillance_llm.generate_event_description(event, api_key)
        event_descriptions.append(description)
    
    # Step 5: Generate overall summary
    print("Generating video summary...")
    summary = surveillance_llm.generate_summary(event_descriptions, video_info, api_key)
    
    # Step 6: Export events to file
    print("Exporting events...")
    events_output_path = os.path.join(output_dir, "events.json")
    event_processor.export_events(events, events_output_path)
    
    # Step 7: Save summary to file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    
    # Step 8: Create Streamlit app for interactive exploration
    print("Creating interactive app...")
    app_path = os.path.join(output_dir, "app.py")
    create_streamlit_app(video_path, frames_dir, events_output_path, summary_path, app_path)
    
    print(f"Processing complete. Results saved to {output_dir}")
    print(f"To launch the interactive app, run: streamlit run {app_path}")
    
    return {
        "video_info": video_info,
        "events": events,
        "summary": summary,
        "output_dir": output_dir
    }

if __name__ == "__main__":
    streamlit_app = create_streamlit_app()
    
    parser = argparse.ArgumentParser(description="Process a video file")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    process_surveillance_video(args.video_path)