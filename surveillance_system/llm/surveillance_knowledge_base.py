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

# ========== 6. KNOWLEDGE BASE ==========

class SurveillanceKnowledgeBase:
    def __init__(self, api_key, index_name="surveillance-events"):
        """
        Initialize the vector database for storing event information
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the index to use
        """
        # Initialize Pinecone
        pinecone.init(api_key=api_key)
        
        # Check if index exists, create if not
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=384,  # Dimension of sentence-transformers model
                metric="cosine"
            )
        
        # Connect to index
        self.index = pinecone.Index(index_name)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_event(self, event_id, event_data, description):
        """
        Add an event to the knowledge base
        
        Args:
            event_id: Unique ID for the event
            event_data: Event dictionary
            description: Natural language description of the event
        """
        # Create metadata from event data
        metadata = {
            'type': event_data['type'],
            'timestamp': str(event_data['timestamp']),
            'frame_idx': int(event_data['frame_idx']),
            'class_name': event_data.get('class_name', ''),
            'confidence': float(event_data.get('confidence', 0)),
            'description': description
        }
        
        # Generate embedding for the description
        embedding = self.embedding_model.encode(description).tolist()
        
        # Upsert to Pinecone
        self.index.upsert(
            vectors=[(str(event_id), embedding, metadata)]
        )
    
    def query_events(self, query_text, top_k=5):
        """
        Query the knowledge base for events matching the query
        
        Args:
            query_text: Natural language query
            top_k: Number of results to return
        
        Returns:
            List of matching events with similarity scores
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results