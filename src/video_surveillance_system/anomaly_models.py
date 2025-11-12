"""
AI model wrappers for anomaly detection.
"""

import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from ultralytics import YOLO  # type: ignore
import config


class AnomalyDetector:
    """Unified detector for weapons and suspicious motion."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load YOLO for weapon detection
        if not config.YOLO_MODEL_PATH or not config.MODEL_A_PATH:
            raise ValueError("YOLO_MODEL_PATH is not set in configuration.")
        print(f"Loading YOLO model from {config.YOLO_MODEL_PATH}...")

        self.yolo_model = YOLO(config.YOLO_MODEL_PATH)

        # Load VideoMAE for motion analysis
        print("Loading VideoMAE model...")
        binary_class_names = ["Normal", "Anomaly"]
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.motion_model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base",
            num_labels=len(binary_class_names),
            ignore_mismatched_sizes=True,
        ).to(self.device) # type: ignore

        self.motion_model.load_state_dict(
            torch.load(config.MODEL_A_PATH, map_location=self.device)
        )
        self.motion_model.eval()

        print("✅ All models loaded successfully.")

    def detect_weapons(self, frame):
        """
        Detect weapons in a single frame.
        Returns: (weapon_found: bool, weapon_class: str or None)
        """
        results = self.yolo_model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                confidence = box.conf.item()

                if (
                    class_name in config.WEAPON_CLASSES
                    and confidence >= config.CONFIDENCE_THRESHOLD_OBJECT
                ):
                    return True, class_name

        return False, None

    def detect_suspicious_motion(self, frames_pil):
        """
        Analyze a sequence of PIL frames for suspicious motion.
        Args:
            frames_pil: List of PIL Image objects (should be NUM_FRAMES in length)
        Returns: (is_suspicious: bool, confidence: float)
        """
        if len(frames_pil) != config.NUM_FRAMES:
            return False, 0.0

        with torch.no_grad():
            inputs = self.processor(frames_pil, return_tensors="pt").to(self.device)
            outputs = self.motion_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

            is_anomaly = torch.argmax(probs).item() == 1
            confidence = probs[1].item()

            is_suspicious = (
                is_anomaly and confidence >= config.CONFIDENCE_THRESHOLD_MOTION
            )

            return is_suspicious, confidence

