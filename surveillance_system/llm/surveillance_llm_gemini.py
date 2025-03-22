import os
import time
from typing import List, Dict, Any
from google import genai

class SurveillanceGeminiLLM:
    def __init__(self, api_key, model="gemini-2.0-flash"):
        """
        Initialize the LLM integration with Google Gemini
        
        Args:
            api_key: API key for Google Gemini
            model: Model to use (default: gemini-2.0-flash)
        """
        # genai.configure(api_key=api_key)
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.system_prompt = """
        You are an AI assistant analyzing surveillance footage. Your task is to:
        1. Summarize events described in surveillance footage with precision and clarity
        2. Identify potential security concerns or suspicious activities
        3. Provide factual descriptions without speculation
        4. Focus on observable behaviors and patterns
        5. Use neutral, professional language
        
        Format your responses concisely with timestamps, locations, and event descriptions with objects and person(s) if involved.
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
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    {"role": "user", "parts": [{"text": self.system_prompt + "\n\n" + prompt}]}
                ]
            )
            
            return response.candidates[0].content.parts[0].text
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
            prompt = "Please describe each of the following surveillance events concisely in a SINGLE LINE by observing the relation between the objects present in the event. Format your response with EVENT 1: [description], EVENT 2: [description], etc. Focus on the most important aspects like detected object, anomaly status, and priority.\n\n"
            
            for idx, event in enumerate(batch):
                event_time = str(event['timestamp'])
                prompt += f"EVENT {idx+1}:\n"
                prompt += f"- Time: {event_time}\n"
                prompt += f"- Type: {event['type']}\n"
                prompt += f"- Subtype: {event.get('subtype', 'N/A')}\n"
                prompt += f"- Object: {event.get('class_name', 'Unknown')}\n"
                prompt += f"- Confidence: {event.get('confidence', 'N/A')}\n"
                prompt += f"- Position: {event.get('bbox', 'N/A')}\n"
                prompt += f"- Priority: {event.get('priority', 'N/A')}\n\n"
            
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        {"role": "user", "parts": [{"text": "You are a surveillance system analyst. Your task is to describe security events concisely and professionally in a single line per event.\n\n" + prompt}]}
                    ]
                )
                
                result_text = response.candidates[0].content.parts[0].text
                
                # More robust parsing approach
                # First, check if the response contains "EVENT" markers
                if "EVENT" in result_text:
                    # Try to extract descriptions using EVENT markers
                    for idx, event in enumerate(batch):
                        event_marker = f"EVENT {idx+1}:"
                        next_marker = f"EVENT {idx+2}:" if idx < len(batch)-1 else None
                        
                        if event_marker in result_text:
                            start_pos = result_text.find(event_marker) + len(event_marker)
                            end_pos = result_text.find(next_marker) if next_marker and next_marker in result_text else None
                            
                            description = result_text[start_pos:end_pos].strip() if end_pos else result_text[start_pos:].strip()
                            # Remove any bullet points, asterisks or newlines
                            description = description.replace('*', '').replace('\n', ' ').strip()
                            descriptions[id(event)] = description
                        else:
                            # Fallback for this specific event
                            event_time = str(event['timestamp'])
                            obj_type = event.get('class_name', 'Unknown')
                            event_type = event['type']
                            priority = event.get('priority', 'N/A')
                            descriptions[id(event)] = f"{event_time}: {event_type.capitalize()} {obj_type} detected with {priority} priority."
                else:
                    # Alternative parsing approach for bulleted or asterisk lists
                    lines = [line.strip() for line in result_text.split('\n') if line.strip()]
                    description_lines = [line for line in lines if line.startswith('*') or line[0].isdigit() or ':' in line]
                    
                    # Match descriptions to events
                    for idx, event in enumerate(batch):
                        if idx < len(description_lines):
                            # Clean up the description (remove bullets, asterisks)
                            description = description_lines[idx].lstrip('*').lstrip('0123456789.').strip()
                            descriptions[id(event)] = description
                        else:
                            # Fallback if we don't have enough descriptions
                            event_time = str(event['timestamp'])
                            obj_type = event.get('class_name', 'Unknown')
                            event_type = event['type']
                            priority = event.get('priority', 'N/A')
                            descriptions[id(event)] = f"{event_time}: {event_type.capitalize()} {obj_type} detected with {priority} priority."
                
            except Exception as e:
                print(f"Error generating batch descriptions: {e}")
                # Fallback: provide simple descriptions
                for event in batch:
                    event_time = str(event['timestamp'])
                    obj_type = event.get('class_name', 'Unknown')
                    event_type = event['type']
                    priority = event.get('priority', 'N/A')
                    descriptions[id(event)] = f"{event_time}: {event_type.capitalize()} {obj_type} detected with {priority} priority."
        
        return descriptions
    
    def generate_events_summary(self, event_descriptions: List, time_period="entire video"):
        """
        Generate a summary of all events within a given time frame
        
        Args:
            event_descriptions: Dictionary mapping event IDs to descriptions
            time_period: Description of the time period (default is "entire video")
        
        Returns:
            A string summary of the events
        """
        # Compile all descriptions into a single prompt
        descriptions_text = "\n".join([f"- {desc}" for desc in event_descriptions])
        
        prompt = f"""As a surveillance system analyst, create a concise summary of the following events detected during the {time_period}. 
    Focus on patterns, significant events, anomalies, and high-priority incidents. 
    Provide an overall assessment of the surveillance period.

    EVENTS:
    {descriptions_text}

    Please provide:
    1. A concise paragraph summarizing the key findings
    2. Any patterns or trends observed
    3. Highlight of high-priority or anomalous events
    """
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    {"role": "user", "parts": [{"text": prompt}]}
                ]
            )
            
            summary = response.candidates[0].content.parts[0].text
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Summary generation failed: {e}"
    
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
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    {"role": "user", "parts": [{"text": self.system_prompt + "\n\n" + prompt}]}
                ]
            )
            
            return response.candidates[0].content.parts[0].text
        except Exception as e:
            print(f"Error answering query: {e}")
            return "I'm sorry, I couldn't process that query about the surveillance footage."
        
    def test_llm(self):
        print("Testing the connection with the Gemini LLM")
        prompt = "Hello How are you doing today"
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    {"role": "user", "parts": [{"text": "You are a surveillance system analyst. Your task is to describe security events concisely and professionally.\n\n" + prompt}]}
                ]
            )
            print(response.candidates[0].content.parts[0].text)
        except Exception as e:
            print(f"Error testing Gemini connection: {e}")
    
    def generate_event_description_with_actions(self, event):
        """
        Generate a description of an event that includes action recognition information.
        
        Args:
            event: Event data including action recognition results
            
        Returns:
            Generated description
        """
        # Extract basic event info
        event_type = event["type"]
        start_time = event["start_time"]
        objects = event["objects"]
        
        # Extract action info if available
        action_info = ""
        if "actions" in event and event["actions"]:
            primary_action = event["primary_action"]
            confidence = event["action_confidence"]
            
            if confidence > 0.7:
                action_info = f"The subject is {primary_action}. "
            elif confidence > 0.5:
                action_info = f"The subject appears to be {primary_action}. "
            else:
                action_info = f"The subject might be {primary_action}, but this is uncertain. "
                
            # Add secondary actions if available
            secondary_actions = []
            for action in event["actions"][1:3]:  # Top 2 secondary actions
                if action["confidence"] > 0.4:
                    secondary_actions.append(action["action"])
                    
            if secondary_actions:
                action_info += f"Other potential activities include {', '.join(secondary_actions)}. "
        
        # Craft the prompt
        prompt = f"""
        Generate a concise description of a surveillance event.
        
        Event Type: {event_type}
        Time: {start_time}
        Objects Detected: {', '.join(objects)}
        {action_info}
        
        Please describe this surveillance event in a factual, neutral manner.
        """
        
        # Use regular LLM generation with the enhanced prompt
        return self.client.messages.create(
            model=self.model,
            max_tokens=150,
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a surveillance system assistant that describes events objectively."},
                {"role": "user", "content": prompt}
            ]
        ).content
        
    def generate_batch_event_descriptions_with_actions(self, events, batch_size=10):
        """
        Generate descriptions for a batch of events with action recognition.
        
        Args:
            events: List of events with action recognition
            batch_size: Number of events to process at once
            
        Returns:
            Dictionary mapping event IDs to descriptions
        """
        descriptions = {}
        
        # Process in batches
        for i in range(0, len(events), batch_size):
            batch = events[i:i+batch_size]
            
            # Process each event in batch
            for event in batch:
                description = self.generate_event_description_with_actions(event)
                descriptions[id(event)] = description
                
        return descriptions

    def generate_events_summary_with_actions(self, event_descriptions, action_stats=None):
        """
        Generate a summary of all events, incorporating action recognition.
        
        Args:
            event_descriptions: List of event descriptions
            action_stats: Optional statistics about actions detected
            
        Returns:
            Summary of events
        """
        descriptions_text = "\n".join([f"- {desc}" for desc in event_descriptions])
        
        action_info = ""
        if action_stats:
            action_counts = action_stats.get("action_counts", {})
            if action_counts:
                top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                action_info = "Common activities detected: " + ", ".join([f"{action} ({count} times)" for action, count in top_actions])
        
        prompt = f"""
        Generate a comprehensive summary of surveillance footage based on the following event descriptions:
        
        {descriptions_text}
        
        {action_info}
        
        Please summarize:
        1. Key events and patterns
        2. Notable subjects or objects
        3. Any suspicious or unusual activity
        4. Overall assessment of the surveillance period
        """
        
        return self.client.messages.create(
            model=self.model,
            max_tokens=500,
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a surveillance analysis assistant that summarizes events objectively."},
                {"role": "user", "content": prompt}
            ]
        ).content