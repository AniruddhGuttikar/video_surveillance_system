"""
Gemini AI service for video analysis and query answering.
"""

import os
import json
import time
import mimetypes
from google import genai
from google.genai.types import Part
import config

INLINE_THRESHOLD = 10 * 1024 * 1024  # 10 MB

SYSTEM_PROMPT = (
    "You are a surveillance security analyst. "
    "Your task is to watch and analyze the provided video footage in its entirety. "
    "Provide a detailed structured JSON array of objects, where each object has exactly these keys:\n"
    "timestamp (string, format HH:MM:SS), severity (string: INFO, WARN, ALERT), summary (string).\n\n"
    "Summary should be detailed and should include specific actions, events, or behaviors observed in the video without missing nuances.\n"
    "Severity classification rules:\n"
    "- INFO: Routine or unremarkable activity.\n"
    "- WARN: Unusual, suspicious, or potentially concerning behavior.\n"
    "- ALERT: Critical events requiring immediate attention or action.\n\n"
    "Example:\n"
    "[\n"
    '  {"timestamp": "00:01:12", "severity": "ALERT", "summary": "Intruder jumps over fence, wearing black colored hoodie"},\n'
    '  {"timestamp": "00:02:45", "severity": "WARN", "summary": "Suspicious blue car circling area with 3 members inside it."}\n'
    "]\n\n"
    "Where the timestamp = timestamp + (Start Timestamp). Fetch (Start Timestamp) from the metadata containing: '(Clip Name): (Start Timestamp) - (End Timestamp). Reason: (Detected Anomaly)'\n\n"
    "Return only a valid JSON array. Do not include any extra text, formatting, or code fences."
)


class GeminiAnalyzer:
    """Handles video analysis and query answering using Gemini."""

    def __init__(self, model="gemini-2.5-flash"):
        self.model = model
        api_key = config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        print(f"✅ Gemini client initialized with model: {model}")

    def analyze_clip(self, video_path, metadata=""):
        """
        Analyze a video clip and return structured events.

        Args:
            video_path: Path to the video file
            metadata: Additional context about the clip

        Returns:
            list[dict]: List of event dictionaries with timestamp, severity, summary
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        mime = mimetypes.guess_type(video_path)[0]
        if not mime:
            mime = "video/mp4"

        # Upload or inline the video
        size = os.path.getsize(video_path)
        if size <= INLINE_THRESHOLD:
            with open(video_path, "rb") as f:
                video_part = Part.from_bytes(data=f.read(), mime_type=mime)
        else:
            uploaded = self.client.files.upload(file=video_path)
            while uploaded.name:
                file_info = self.client.files.get(name=uploaded.name)
                if file_info.state == "ACTIVE":
                    break
                elif file_info.state == "FAILED":
                    raise RuntimeError(f"File {uploaded.name} failed to process")
                time.sleep(1)
            video_part = uploaded

        # Create metadata instruction
        metadata_instruction = (
            f"Additional metadata for this video: {metadata}\n"
            "Please take this into account when producing the structured summary.\n"
            "It contains: '(Clip Name): (Start Timestamp) - (End Timestamp). Reason: (Detected Anomaly)'"
        )

        # Generate analysis
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                SYSTEM_PROMPT,
                metadata_instruction,
                video_part,
            ],
        )

        raw_text = getattr(response, "text", None) or str(response)

        # Parse JSON response
        try:
            events = json.loads(raw_text)
            return events
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse Gemini response as JSON: {e}")
            print(f"Raw output: {raw_text}")
            return []

    def answer_query(self, query, search_results):
        """
        Answer a user query based on search results from the vector database.

        Args:
            query: User's question
            search_results: List of relevant events from vector search

        Returns:
            str: Answer to the query
        """
        if not search_results:
            return "No relevant events found in the surveillance logs."

        # Build context from search results
        events_context = "\n".join(
            f"[{item['timestamp']}] ({item['severity']}): {item['summary']}"
            for item in search_results
        )

        prompt = (
            "You are an investigative analyst reviewing structured surveillance event logs.\n"
            "Each log entry has a timestamp, severity level, and a short summary.\n"
            "Answer the user's question concisely and directly using only the given event summaries.\n"
            "If the answer cannot be found, say 'Information not available in the provided events.'\n\n"
            f"Events:\n{events_context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        response = self.client.models.generate_content(
            model=self.model, contents=[prompt]
        )

        return getattr(response, "text", "").strip()
