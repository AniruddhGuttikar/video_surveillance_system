"""
Vector database operations using Qdrant for semantic search.
"""

import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import google.genai as genai
import config


class VectorDB:
    """Manages surveillance event storage and retrieval in Qdrant."""

    def __init__(self, collection_name="surveillance_logs"):
        self.collection_name = collection_name

        # Validate configuration
        if not all([config.QDRANT_URL, config.QDRANT_API_KEY, config.GEMINI_API_KEY]):
            raise ValueError(
                "Missing required environment variables: QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY"
            )

        # Initialize clients
        self.client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        self.genai_client = genai.Client(api_key=config.GEMINI_API_KEY)

        self._ensure_collection()
        print(f"✅ Vector DB initialized with collection: {collection_name}")

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            print(f"Attempting to recreate collection: {self.collection_name}")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "embedding": VectorParams(size=768, distance=Distance.COSINE)
                },
            )
            print(f"Created collection: {self.collection_name}")

        except Exception as e:
            print(f"Error ensuring collection: {e}")
            raise

    def _embed_text(self, text):
        """Generate embeddings using Google's text-embedding-004 model."""
        try:
            result = self.genai_client.models.embed_content(
                model="text-embedding-004", contents=text
            )
            if not result.embeddings or not result.embeddings[0].values:
                raise ValueError("Embedding generation failed")
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def add_events(self, video_id, events):
        """
        Add multiple events from a video to the database.

        Args:
            video_id: Identifier for the source video
            events: List of dicts with keys: timestamp, severity, summary

        Returns:
            int: Number of events added
        """
        points = []

        for event in events:
            try:
                # Generate embedding
                vector = self._embed_text(event["summary"])

                # Create payload
                payload = {
                    "video_id": video_id,
                    "timestamp": event["timestamp"],
                    "severity": event["severity"],
                    "summary": event["summary"],
                }

                # Create unique point ID
                id_string = f"{video_id}_{event['timestamp']}_{event['summary'][:50]}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_string))

                points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            except Exception as e:
                print(f"Warning: Failed to process event: {e}")
                continue

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        return len(points)

    def search(self, query_text, top_k=5):
        """
        Search for events semantically similar to the query.

        Args:
            query_text: Search query
            top_k: Number of results to return

        Returns:
            list[dict]: Search results with score, video_id, timestamp, severity, summary
        """
        try:
            # Generate query embedding
            query_vector = self._embed_text(query_text)

            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            # Format results
            formatted_results = []
            for result in results:
                if result.payload:
                    formatted_results.append(
                        {
                            "id": result.id,
                            "score": result.score,
                            "video_id": result.payload.get("video_id"),
                            "timestamp": result.payload.get("timestamp"),
                            "severity": result.payload.get("severity"),
                            "summary": result.payload.get("summary"),
                        }
                    )

            return formatted_results

        except Exception as e:
            print(f"Error searching: {e}")
            return []
