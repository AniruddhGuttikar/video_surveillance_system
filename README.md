# Video Anomaly Detection System

AI-powered video surveillance system that detects weapons and suspicious motion, analyzes events with Gemini, and enables semantic search of incidents.

## Features

- **Dual Detection**: YOLO for weapons, VideoMAE for suspicious motion
- **AI Analysis**: Gemini provides detailed event summaries
- **Semantic Search**: Query incidents using natural language via Qdrant
- **Real-time Monitoring**: Live detection with SMS alerts
- **Batch Processing**: Analyze uploaded videos and extract clips

## Project Structure

````
anomaly-detection-system/
├── pyproject.toml          # uv project configuration
├── .env                    # Environment variables (create from .env.example)
├── src/
│   └── video_surveillance_system/
│       ├── config.py               # All configuration constants
│       ├── anomaly_models.py       # YOLO + VideoMAE wrapper
│       ├── gemini_service.py       # Gemini video analysis
│       ├── vector_db.py            # Qdrant operations
│       └── app.py                  # Main Flask application
├── uploads/                # Uploaded videos
├── clips/                  # Extracted anomaly clips
├── temp/                   # Temporary processing files
└── models/                 # Anomaly models



## Setup

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

### 2. Create project and install dependencies

```bash
# Initialize with uv
uv sync

# Or manually install dependencies
uv add flask flask-cors opencv-python torch torchvision transformers ultralytics moviepy twilio qdrant-client google-genai python-dotenv pillow numpy tqdm
```

### 3. Download required models

Place these files in models directory:

- `Guns-100-11m.pt` - YOLO weapon detection model
- `videomae_model_A_binary.pth` - VideoMAE motion detection model

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:

- **Twilio**: Account SID, Auth Token, Phone Number
- **Qdrant**: Cloud URL and API key
- **Gemini**: Google AI API key

### 5. Run the application

```bash
uv run src/video_surveillance_system/app.py
```

The server will start on `http://0.0.0.0:5001`

## API Endpoints

### Video Processing

**POST /upload**

- Upload video for batch analysis
- Form data: `video` (file)

**POST /start_simulation**

- Start real-time monitoring
- Form data: `demo_video` (file), `phone_number` (string)

**POST /stop_realtime**

- Stop real-time monitoring

### Results & Monitoring

**GET /video_feed**

- Live video stream (MJPEG)

**GET /status**

- Current system status

**GET /results**

- List all extracted clips

**GET /results/<filename>**

- Download specific clip

### Chat Interface

**POST /chat**

- Query detected events
- JSON body: `{"question": "your question here"}`
- Returns: `{"answer": "..."}`

## How It Works

### Batch Processing Flow

1. **Upload Video** → Video saved to `uploads/`
2. **Motion Preprocessing** → Remove static frames
3. **Anomaly Detection** → Run YOLO + VideoMAE on frames
4. **Event Merging** → Combine overlapping detections
5. **Clip Extraction** → Save anomaly segments to `clips/`
6. **Gemini Analysis** → Generate detailed event summaries
7. **Vector Storage** → Store embeddings in Qdrant

### Real-time Monitoring Flow

1. **Frame Capture** → Read frames from video source
2. **Detection** → Check each frame for anomalies
3. **Debouncing** → Confirm anomaly across multiple frames
4. **Recording** → Save clip with pre/post-roll
5. **SMS Alert** → Notify via Twilio
6. **Streaming** → Live feed available at `/video_feed`

### Query System

1. User asks question via `/chat`
2. Question embedded using Gemini embeddings
3. Semantic search in Qdrant vector DB
4. Relevant events retrieved
5. Gemini generates answer from context

## Configuration

Edit `config.py` to adjust:

- **Detection thresholds**: `CONFIDENCE_THRESHOLD_MOTION`, `CONFIDENCE_THRESHOLD_OBJECT`
- **Real-time parameters**: `DEBOUNCE_COUNT`, `BUFFER_SECONDS`, `POST_ROLL_FRAMES`
- **Processing settings**: `NUM_FRAMES`, `REALTIME_FRAME_SKIP`, `TARGET_CLIP_SIZE_MB`

## Notes

- GPU recommended for real-time performance
- Clips are saved as H.264 MP4 files
- Vector DB persists across sessions
- SMS alerts have 60-second cooldown by default
