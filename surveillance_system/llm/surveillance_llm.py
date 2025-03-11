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


# ========== 5. LLM INTEGRATION ==========

class SurveillanceLLM:
    def __init__(self, api_key, model="claude-3.5-sonnet-20240307"):
        """
        Initialize the LLM integration
        
        Args:
            api_key: API key for Claude or other LLM
            model: Model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.system_prompt = """
        You are an AI assistant analyzing surveillance footage. Your task is to:
        1. Summarize events described in surveillance footage with precision and clarity
        2. Identify potential security concerns or suspicious activities
        3. Provide factual descriptions without speculation
        4. Focus on observable behaviors and patterns
        5. Use neutral, professional language
        
        Format your responses concisely with timestamps, locations, and event descriptions.
        """
        
    def generate_event_description(self, event, context_events=[]):
        """
        Generate a natural language description of an event
        
        Args:
            event: Event data dictionary
            context_events: List of related events for context
        
        Returns:
            String description of the event
        """
        # Create a prompt describing the event
        event_time = str(event['timestamp'])
        
        prompt = f"""
        Please describe the following surveillance event detected at {event_time}:
        
        Event type: {event['type']}
        Object detected: {event.get('class_name', 'Unknown')}
        Confidence: {event.get('confidence', 'N/A')}
        """
        
        if context_events:
            prompt += "\n\nContext (other related events):\n"
            for ctx_event in context_events[:5]:  # Limit to 5 for brevity
                ctx_time = str(ctx_event['timestamp'])
                prompt += f"- {ctx_time}: {ctx_event['type']} ({ctx_event.get('class_name', 'Unknown')})\n"
        
        try:
            response = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error generating event description: {e}")
            return f"Event: {event['type']} at {event_time}"
    
    def generate_batch_event_descriptions(self, events, batch_size=10):
        """
        Generate descriptions for multiple events in a single API call
        
        Args:
            events: List of event dictionaries
            batch_size: Number of events to process in each API call
        
        Returns:
            Dictionary mapping event IDs to descriptions
        """
        descriptions = {}
        
        # Process events in batches
        for i in range(0, len(events), batch_size):
            batch = events[i:i+batch_size]
            
            # Create a comprehensive prompt for all events in the batch
            prompt = "Please describe each of the following surveillance events concisely:\n\n"
            
            for idx, event in enumerate(batch):
                event_time = str(event['timestamp'])
                prompt += f"EVENT {idx+1}:\n"
                prompt += f"- Time: {event_time}\n"
                prompt += f"- Type: {event['type']}\n"
                prompt += f"- Subtype: {event.get('subtype', 'N/A')}\n"
                prompt += f"- Object: {event.get('class_name', 'Unknown')}\n"
                prompt += f"- Confidence: {event.get('confidence', 'N/A')}\n"
                prompt += f"- Position: {event.get('bbox', 'N/A')}\n\n"
            
            prompt += "For each event, provide a one-sentence description in the format 'EVENT 1: [description]'"
            
            try:
                # Use Claude Sonnet which has lower token usage
                response = self.client.messages.create(
                    model=self.model,
                    system="You are a surveillance system analyst. Your task is to describe security events concisely and professionally.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=batch_size * 75
                )
                
                result_text = response.content[0].text
                
                # Parse the response to extract individual descriptions
                for idx, event in enumerate(batch):
                    event_marker = f"EVENT {idx+1}:"
                    next_marker = f"EVENT {idx+2}:" if idx < len(batch)-1 else None
                    
                    if event_marker in result_text:
                        start_pos = result_text.find(event_marker) + len(event_marker)
                        end_pos = result_text.find(next_marker) if next_marker else None
                        
                        description = result_text[start_pos:end_pos].strip()
                        descriptions[id(event)] = description
                    else:
                        descriptions[id(event)] = f"Event: {event['type']} at {event_time}"
                
                # Respect rate limits - wait if necessary
                time.sleep(12 / batch_size)  # To stay under 5 requests per minute
                
            except Exception as e:
                print(f"Error generating batch descriptions: {e}")
                # Fallback: provide simple descriptions
                for event in batch:
                    event_time = str(event['timestamp'])
                    descriptions[id(event)] = f"Event: {event['type']} at {event_time}"
        
        return descriptions
    
    def generate_summary(self, events, time_period="the entire recording"):
        """
        Generate a summary of multiple events
        
        Args:
            events: List of event dictionaries
            time_period: Description of the time period
        
        Returns:
            String summary of the events
        """
        if not events:
            return f"No events detected during {time_period}."
        
        # Count events by type
        event_counts = {}
        for event in events:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Create a summary of event counts
        counts_text = "\n".join([f"- {count} instances of {event_type}" 
                                 for event_type, count in event_counts.items()])
        
        # Create a chronological list of significant events
        significant_events = [e for e in events if e.get('priority') == 'high']
        if not significant_events:
            # If no high priority events, take a sample of regular events
            significant_events = events[:min(5, len(events))]
        
        events_text = "\n".join([f"- At {e['timestamp']}: {e['type']} ({e.get('class_name', 'Unknown')})" 
                                 for e in significant_events])
        
        prompt = f"""
        Please provide a surveillance summary for {time_period}.
        
        Event statistics:
        {counts_text}
        
        Significant events:
        {events_text}
        
        Please provide:
        1. A brief overview of the activity level
        2. Key events that occurred
        3. Any patterns or anomalies worth noting
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Summary of {len(events)} events during {time_period}."
    
    def answer_query(self, query, events, video_info):
        """
        Answer a natural language query about the surveillance footage
        
        Args:
            query: User's question as string
            events: List of events to reference
            video_info: Dictionary with video metadata
        
        Returns:
            String response to the query
        """
        # Extract key information from events for context
        event_summary = "\n".join([
            f"- At {e['timestamp']}: {e['type']} ({e.get('class_name', 'Unknown')})" 
            for e in events[:20]  # Limit to 20 for brevity
        ])
        
        video_context = f"""
        Video information:
        - Filename: {video_info.get('filename', 'Unknown')}
        - Duration: {video_info.get('duration', 'Unknown')} seconds
        - Start time: {video_info.get('start_time', 'Unknown')}
        """
        
        prompt = f"""
        Please answer the following question about surveillance footage:
        
        User question: {query}
        
        {video_context}
        
        Events detected:
        {event_summary}
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error answering query: {e}")
            return "I'm sorry, I couldn't process that query about the surveillance footage."
