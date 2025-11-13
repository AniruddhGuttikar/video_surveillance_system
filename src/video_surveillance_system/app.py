"""
Main application: Video anomaly detection system with AI analysis.
CHANGES FOR UNIFIED PROCESSING:
1. Track video timestamps (not wall-clock) in real-time mode
2. Send all clips to Gemini + vector DB (both modes)
3. Consistent timestamp format across both modes
"""

import os
import cv2
import time
import numpy as np
import threading
import queue
from collections import deque
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from moviepy import VideoFileClip
from flask import Flask, Response, request, jsonify, send_from_directory
from flask_cors import CORS
from twilio.rest import Client

import config
from anomaly_models import AnomalyDetector
from gemini_service import GeminiAnalyzer
from vector_db import VectorDB

# --- GLOBAL STATE ---
real_time_status = "System Standby"
camera_frame = None
stop_real_time_thread = threading.Event()
real_time_target_phone = None
real_time_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=128)

# Log storage for frontend
event_logs = []
event_logs_lock = threading.Lock()

# --- INITIALIZE AI SERVICES ---
print("🚀 Initializing AI services...")
detector = AnomalyDetector()
gemini = GeminiAnalyzer()
vector_db = VectorDB()

# --- SHARED FUNCTION FOR CLIP ANALYSIS ---


def analyze_and_store_clip(clip_filename, start_time, end_time, reason, clip_counter):
    """
    Shared function to analyze clip with Gemini and store in vector DB.
    Works for both uploaded and real-time clips.

    Args:
        clip_filename: Path to the clip file
        start_time: Start time in seconds (video time)
        end_time: End time in seconds (video time)
        reason: Detected anomaly reason
        clip_counter: Clip number for logging

    Returns:
        int: Number of events added to database
    """
    # Format timestamps as HH:MM:SS
    start_str = time.strftime("%H:%M:%S", time.gmtime(start_time))
    end_str = time.strftime("%H:%M:%S", time.gmtime(end_time))

    # Prepare metadata for Gemini
    metadata = (
        f"{os.path.basename(clip_filename)}: {start_str} - {end_str}. Reason: {reason}"
    )

    try:
        # Analyze with Gemini
        events = gemini.analyze_clip(clip_filename, metadata)

        if events:
            # Add to vector database
            count = vector_db.add_events(clip_filename, events)

            # Save events to global log for frontend
            with event_logs_lock:
                for event in events:
                    event_logs.append(
                        {
                            "video_id": clip_filename,
                            "clip_number": clip_counter,
                            "timestamp": event["timestamp"],
                            "severity": event["severity"],
                            "summary": event["summary"],
                        }
                    )

            print(f"✅ Clip {clip_counter}: Added {count} events to database")
            return count
        else:
            print(f"⚠️ Clip {clip_counter}: No events detected")
            return 0

    except Exception as e:
        print(f"❌ Failed to analyze clip {clip_counter}: {e}")
        return 0


# --- VIDEO PROCESSING FUNCTIONS ---


def preprocess_video_for_motion(video_path):
    """
    Remove static frames from video and return processed video path + timestamp map.
    """
    print(f"Preprocessing video: {video_path}")
    os.makedirs(config.TEMP_FOLDER, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output_path = os.path.join(
        config.TEMP_FOLDER, f"motion_{os.path.basename(video_path)}"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    timestamp_map = []
    ret, prev_frame = cap.read()
    if not ret:
        print("Warning: Could not read first frame")
        cap.release()
        out.release()
        return None, None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    out.write(prev_frame)
    timestamp_map.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, gray)
        motion_score = np.mean(frame_diff)

        if motion_score > config.STATIC_FRAME_MOTION_THRESHOLD:
            out.write(frame)
            timestamp_map.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

        prev_gray = gray

    cap.release()
    out.release()
    print(f"✅ Preprocessing complete: {len(timestamp_map)} motion frames")
    return temp_output_path, timestamp_map


def get_video_bitrate(video_path):
    """Estimate video bitrate in bits per second."""
    file_size_bytes = os.path.getsize(video_path)
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return (file_size_bytes * 8) / duration if duration > 0 else 2000000


def process_uploaded_video(video_path):
    """
    Main processing pipeline for uploaded videos.
    Detects anomalies, extracts clips, analyzes with Gemini, stores in vector DB.
    """
    global real_time_status

    # Step 1: Preprocess for motion
    with real_time_lock:
        real_time_status = "Step 1/4: Pre-processing for motion..."

    motion_video_path, timestamp_map = preprocess_video_for_motion(video_path)
    if not motion_video_path:
        with real_time_lock:
            real_time_status = "Error: Could not process video."
        return

    # Step 2: Detect anomalies
    with real_time_lock:
        real_time_status = "Step 2/4: Detecting anomalies..."

    detected_events = []
    cap = cv2.VideoCapture(motion_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_index in tqdm(range(total_frames), desc="Analyzing Frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        # Check for weapons
        weapon_found, weapon_class = detector.detect_weapons(frame)
        if weapon_found:
            timestamp = timestamp_map[frame_index]
            detected_events.append(
                (timestamp, timestamp + 1.0, f"Weapon ({weapon_class})")
            )

        # Check for suspicious motion every half second
        if (
            frame_index % int(fps / 2) == 0
            and (total_frames - frame_index) >= config.NUM_FRAMES
        ):
            frames_for_motion = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            for _ in range(config.NUM_FRAMES):
                ret_m, frame_m = cap.read()
                if not ret_m:
                    break
                frames_for_motion.append(
                    Image.fromarray(cv2.cvtColor(frame_m, cv2.COLOR_BGR2RGB))
                )

            if len(frames_for_motion) == config.NUM_FRAMES:
                is_suspicious, confidence = detector.detect_suspicious_motion(
                    frames_for_motion
                )
                if is_suspicious:
                    start_time = timestamp_map[frame_index]
                    end_idx = min(
                        frame_index + config.NUM_FRAMES, len(timestamp_map) - 1
                    )
                    end_time = timestamp_map[end_idx]
                    detected_events.append((start_time, end_time, "Suspicious Motion"))

    cap.release()

    # Step 3: Merge overlapping events
    with real_time_lock:
        real_time_status = "Step 3/4: Merging events..."

    if not detected_events:
        with real_time_lock:
            real_time_status = "✅ Processing complete. No anomalies detected."
        os.remove(motion_video_path)
        return

    detected_events.sort(key=lambda x: x[0])
    merged_events = []
    current_start, current_end, current_reason = detected_events[0]
    reason_priority = {"Suspicious Motion": 1, "Weapon": 2}

    for next_start, next_end, next_reason in detected_events[1:]:
        if next_start <= current_end + 2.0:
            current_end = max(current_end, next_end)
            if reason_priority.get(next_reason.split(" ")[0], 0) > reason_priority.get(
                current_reason.split(" ")[0], 0
            ):
                current_reason = next_reason
        else:
            merged_events.append((current_start, current_end, current_reason))
            current_start, current_end, current_reason = (
                next_start,
                next_end,
                next_reason,
            )
    merged_events.append((current_start, current_end, current_reason))

    # Step 4: Extract clips and analyze with Gemini
    with real_time_lock:
        real_time_status = f"Step 4/4: Analyzing {len(merged_events)} clips with AI..."

    original_video = VideoFileClip(video_path)
    video_bitrate = get_video_bitrate(video_path)
    max_duration = (config.TARGET_CLIP_SIZE_MB * 8 * 1024 * 1024) / video_bitrate

    clip_counter = 1
    total_events_added = 0

    for start, end, reason in merged_events:
        duration = end - start
        num_subclips = int(np.ceil(duration / max_duration))

        for i in range(num_subclips):
            sub_start = start + i * max_duration
            sub_end = min(end, sub_start + max_duration)

            if sub_end > original_video.duration:
                continue

            # Extract clip
            clip_filename = os.path.join(
                config.CLIPS_FOLDER, f"anomaly_clip_{clip_counter}.mp4"
            )
            subclip = original_video.subclipped(sub_start, sub_end)
            subclip.write_videofile(
                clip_filename, codec="libx264", audio_codec="aac", logger=None
            )

            # Use shared analysis function
            count = analyze_and_store_clip(
                clip_filename, sub_start, sub_end, reason, clip_counter
            )
            total_events_added += count
            clip_counter += 1

    original_video.close()
    os.remove(motion_video_path)

    with real_time_lock:
        real_time_status = f"✅ Complete: {clip_counter - 1} clips, {total_events_added} events indexed"


# --- REAL-TIME MONITORING ---


def frame_capture_thread(video_source):
    """Capture frames from video source and put them in queue."""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Could not open video source: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_time = 1.0 / fps if fps > 0 else 1.0 / 30

    while not stop_real_time_thread.is_set():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Get current video timestamp (in seconds)
        video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        try:
            frame_queue.put((frame, fps, video_timestamp), timeout=1)
        except queue.Full:
            continue

        # Rate limiting: sleep to match original video FPS
        # elapsed = time.time() - start_time
        # sleep_time = max(0, frame_time - elapsed)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)

    cap.release()
    print("✅ Capture thread finished")


def real_time_monitoring():
    """
    Simplified real-time monitoring with correct event merging and clip recording.
    Records continuously and processes clips after anomalies end.
    """
    global real_time_status, camera_frame

    twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    last_alert_time = -60

    # Detection state
    detected_events = []  # List of (start_time, end_time, reason)
    current_anomaly_start = None
    current_anomaly_reason = None
    debounce_counter = 0

    # Frame buffer for motion detection
    motion_frame_buffer = deque(maxlen=config.NUM_FRAMES)

    # Video recording state
    video_writer = None
    recording_start_time = None
    temp_video_path = None
    frame_counter = 0
    fps = 30  # Default, will be updated

    while not stop_real_time_thread.is_set():
        try:
            frame, fps, video_timestamp = frame_queue.get(timeout=2)
        except queue.Empty:
            break

        frame_counter += 1

        # Update camera feed for streaming
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with real_time_lock:
            camera_frame = buffer.tobytes()

        # Add frame to motion detection buffer
        motion_frame_buffer.append(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        )

        # Always write frame if recording
        if video_writer is not None:
            video_writer.write(frame)

        # Skip detection on some frames for performance
        if frame_counter % config.REALTIME_FRAME_SKIP != 0:
            continue

        # --- DETECTION LOGIC ---
        anomaly_detected = False
        detected_reason = None

        # Check for weapons
        weapon_found, weapon_class = detector.detect_weapons(frame)
        if weapon_found:
            anomaly_detected = True
            detected_reason = f"Weapon ({weapon_class})"

        # Check for suspicious motion (if no weapon found)
        if not weapon_found:
            # Get last N frames for motion detection
            motion_frames = []
            for i in range(config.NUM_FRAMES):
                idx = frame_counter - config.NUM_FRAMES + i
                if idx >= 0:
                    # In real implementation, you'd maintain a frame buffer
                    # For now, assume we can get recent frames
                    motion_frames.append(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    )

            if len(motion_frames) == config.NUM_FRAMES:
                is_suspicious, confidence = detector.detect_suspicious_motion(
                    motion_frames
                )
                if is_suspicious:
                    anomaly_detected = True
                    detected_reason = "Suspicious Motion"

        # --- STATE MANAGEMENT ---
        if anomaly_detected:
            debounce_counter += 1

            # Start new anomaly event
            if (
                current_anomaly_start is None
                and debounce_counter >= config.DEBOUNCE_COUNT
            ):
                current_anomaly_start = video_timestamp
                current_anomaly_reason = detected_reason

                # Start recording if not already recording
                if video_writer is None:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_video_path = os.path.join(
                        config.CLIPS_FOLDER, f"LIVE_recording_{timestamp_str}.mp4"
                    )
                    recording_start_time = video_timestamp

                    height, width, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        temp_video_path, fourcc, fps, (width, height)
                    )

                    with real_time_lock:
                        real_time_status = f"🚨 RECORDING: {detected_reason}"

                # Send SMS alert (with cooldown)
                if (time.time() - last_alert_time) > config.ALERT_COOLDOWN_SECONDS:
                    with real_time_lock:
                        recipient = real_time_target_phone

                    if recipient:
                        try:
                            message_body = (
                                f"🚨 SECURITY ALERT 🚨\n\n"
                                f"Threat: {detected_reason}\n"
                                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            )
                            twilio_client.messages.create(
                                body=message_body,
                                from_=config.TWILIO_PHONE_NUMBER,
                                to=recipient,
                            )
                            print(f"📱 SMS sent to {recipient}")
                            last_alert_time = time.time()
                        except Exception as e:
                            print(f"❌ SMS failed: {e}")

            # Update reason if higher priority detected
            elif current_anomaly_start is not None:
                reason_priority = {"Suspicious Motion": 1, "Weapon": 2}
                current_priority = reason_priority.get(
                    current_anomaly_reason.split(" ")[0], 0
                )
                new_priority = reason_priority.get(detected_reason.split(" ")[0], 0)
                if new_priority > current_priority:
                    current_anomaly_reason = detected_reason

        else:
            # No anomaly detected
            if current_anomaly_start is not None:
                # End current anomaly event
                detected_events.append(
                    (current_anomaly_start, video_timestamp, current_anomaly_reason)
                )
                current_anomaly_start = None
                current_anomaly_reason = None

            debounce_counter = 0

            # Check if we should stop recording (after post-roll period)
            if video_writer is not None:
                time_since_last_event = (
                    video_timestamp - detected_events[-1][1]
                    if detected_events
                    else float("inf")
                )

                if time_since_last_event > config.POST_ROLL_FRAMES / fps:
                    # Stop recording and process
                    video_writer.release()
                    video_writer = None

                    # Process recorded video with event merging
                    if detected_events:
                        process_recorded_clip(
                            temp_video_path, recording_start_time, detected_events
                        )
                        detected_events = []

                    temp_video_path = None
                    recording_start_time = None

                    with real_time_lock:
                        real_time_status = "✅ MONITORING: All Clear"
            else:
                with real_time_lock:
                    real_time_status = "✅ MONITORING: All Clear"

    # Cleanup on exit
    if video_writer is not None:
        video_writer.release()
        if temp_video_path and detected_events:
            process_recorded_clip(
                temp_video_path, recording_start_time, detected_events
            )

    print("✅ Monitoring thread finished")


def process_recorded_clip(video_path, recording_start_time, detected_events):
    """
    Process recorded clip: merge events and extract/analyze clips.
    Based on the working upload video logic.
    """
    # Merge overlapping events (same logic as upload version)
    detected_events.sort(key=lambda x: x[0])
    merged_events = []
    current_start, current_end, current_reason = detected_events[0]
    reason_priority = {"Suspicious Motion": 1, "Weapon": 2}

    for next_start, next_end, next_reason in detected_events[1:]:
        if next_start <= current_end + 2.0:
            current_end = max(current_end, next_end)
            if reason_priority.get(next_reason.split(" ")[0], 0) > reason_priority.get(
                current_reason.split(" ")[0], 0
            ):
                current_reason = next_reason
        else:
            merged_events.append((current_start, current_end, current_reason))
            current_start, current_end, current_reason = (
                next_start,
                next_end,
                next_reason,
            )

    merged_events.append((current_start, current_end, current_reason))

    # Extract and analyze clips
    original_video = VideoFileClip(video_path)
    video_bitrate = get_video_bitrate(video_path)
    max_duration = (config.TARGET_CLIP_SIZE_MB * 8 * 1024 * 1024) / video_bitrate

    clip_counter = 1

    for start, end, reason in merged_events:
        # Adjust times relative to recording start
        clip_start = start - recording_start_time
        clip_end = end - recording_start_time

        # Ensure times are within video bounds
        clip_start = max(0, clip_start)
        clip_end = min(original_video.duration, clip_end)

        duration = clip_end - clip_start
        num_subclips = int(np.ceil(duration / max_duration))

        for i in range(num_subclips):
            sub_start = clip_start + i * max_duration
            sub_end = min(clip_end, sub_start + max_duration)

            if sub_end > original_video.duration:
                continue

            # Extract clip
            clip_filename = os.path.join(
                config.CLIPS_FOLDER,
                f"LIVE_anomaly_clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{clip_counter}.mp4",
            )
            subclip = original_video.subclipped(sub_start, sub_end)
            subclip.write_videofile(
                clip_filename, codec="libx264", audio_codec="aac", logger=None
            )

            # Analyze and store (same as upload version)
            analyze_and_store_clip(
                clip_filename,
                start,  # Use original timestamps for indexing
                end,
                reason,
                clip_counter,
            )
            clip_counter += 1

    original_video.close()

    # Clean up temp recording
    try:
        os.remove(video_path)
    except Exception as e:
        print(f"Warning: Could not remove temp file {video_path}: {e}")


# --- FLASK API ---

app = Flask(__name__)
CORS(app)

# Create folders
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.CLIPS_FOLDER, exist_ok=True)
os.makedirs(config.TEMP_FOLDER, exist_ok=True)


@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload a video for batch processing."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "" or not file.filename:
        return jsonify({"error": "No video file selected"}), 400
    filepath = os.path.join(config.UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    threading.Thread(
        target=process_uploaded_video, args=(filepath,), daemon=True
    ).start()

    return jsonify({"message": f"Started processing {file.filename}"})


@app.route("/start_simulation", methods=["POST"])
def start_simulation():
    """Start real-time monitoring simulation."""
    global stop_real_time_thread, real_time_target_phone

    if "demo_video" not in request.files:
        return jsonify({"error": "No simulation video provided"}), 400

    file = request.files["demo_video"]
    phone_number = request.form.get("phone_number")

    if file.filename == "" or not phone_number or not file.filename:
        return jsonify({"error": "Missing video file or phone number"}), 400

    filepath = os.path.join(config.UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    with real_time_lock:
        real_time_target_phone = phone_number
        stop_real_time_thread.clear()

    if not any(t.name == "analysis_thread" for t in threading.enumerate()):
        threading.Thread(
            target=frame_capture_thread,
            args=(filepath,),
            name="capture_thread",
            daemon=True,
        ).start()

        threading.Thread(
            target=real_time_monitoring, name="analysis_thread", daemon=True
        ).start()

    return jsonify({"message": "Real-time simulation started"})


@app.route("/stop_realtime", methods=["POST"])
def stop_realtime():
    """Stop real-time monitoring."""
    global real_time_status

    with real_time_lock:
        stop_real_time_thread.set()
        real_time_status = "System Standby"

    return jsonify({"message": "Real-time simulation stopping"})


def gen_frames():
    """Generator for streaming video feed."""
    global camera_frame

    while not stop_real_time_thread.is_set():
        time.sleep(1 / 60)

        with real_time_lock:
            if camera_frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Waiting for video stream...",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                _, buffer = cv2.imencode(".jpg", placeholder)
                frame_bytes = buffer.tobytes()
            else:
                frame_bytes = camera_frame

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/video_feed")
def video_feed():
    """Stream live video feed."""
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def get_status():
    """Get current system status."""
    with real_time_lock:
        return jsonify({"status": real_time_status})


@app.route("/results")
def list_results():
    """List all result files."""
    files = [f for f in os.listdir(config.CLIPS_FOLDER) if f.endswith(".mp4")]
    return jsonify({"files": files})


@app.route("/results/<filename>")
def get_result_file(filename):
    """Download a specific result file."""
    return send_from_directory(config.CLIPS_FOLDER, filename)


@app.route("/chat", methods=["POST"])
def chat():
    """Answer questions about detected events."""
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400

    question = data["question"]

    try:
        search_results = vector_db.search(question, top_k=5)
        answer = gemini.answer_query(question, search_results)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logs/count", methods=["GET"])
def get_log_count():
    """Get the count of analyzed clips."""
    with event_logs_lock:
        unique_clips = len(set(log["video_id"] for log in event_logs))
        return jsonify({"count": unique_clips})


@app.route("/logs/all", methods=["GET"])
def get_all_logs():
    """Get all event logs."""
    with event_logs_lock:
        return jsonify({"logs": event_logs})


@app.route("/logs/clip/<int:clip_number>", methods=["GET"])
def get_clip_logs(clip_number):
    """Get logs for a specific clip number."""
    with event_logs_lock:
        clip_events = [log for log in event_logs if log["clip_number"] == clip_number]
        return jsonify({"clip_number": clip_number, "events": clip_events})


if __name__ == "__main__":
    print("🚀 Starting Anomaly Detection System")
    print(f"📁 Uploads: {config.UPLOAD_FOLDER}")
    print(f"📁 Clips: {config.CLIPS_FOLDER}")
    print("🌐 Starting server on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
