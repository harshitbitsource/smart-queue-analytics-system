# Advanced Queue Detection System
## Production-Ready Real-Time Queue Analytics with YOLOv8

A sophisticated computer vision system for real-time queue monitoring and analytics, designed for embedded systems and ML-heavy production environments.

---

## 🎯 Key Features

### ML & Computer Vision
- **YOLOv8 Integration**: Real-time person detection with configurable confidence thresholds
- **Multi-Object Tracking (MOT)**: IoU-based tracking algorithm for person identification and count consistency
- **Confidence-Based Filtering**: Adaptive detection filtering by confidence levels with visual distinction
- **Size-Based Filtering**: Eliminates false positives through height-based bounding box validation
- **Custom Region of Interest (ROI)**: Configurable queue zone detection

### Analytics & Statistics
- **Real-Time Metrics Dashboard**:
  - Current queue length with position tracking
  - Average, max, and min queue lengths
  - Standard deviation analysis
  - Per-person confidence scores
  - Frame rate (FPS) monitoring
  
- **Historical Data Collection**:
  - CSV export with timestamp, queue length, confidence, FPS, and resource metrics
  - JSON statistics summary for post-analysis

### Performance & Resource Management
- **Frame Skipping**: Configurable frame processing for optimized resource utilization
- **System Resource Monitoring**: Real-time CPU and memory usage tracking
- **Performance Optimization**: Threading-ready architecture for multi-threaded inference
- **FPS Tracking**: Real-time performance metrics to identify bottlenecks

### Tracking & Persistence
- **Persistent Person Tracking**: Unique ID assignment to maintain person identity across frames
- **Age-Based Track Management**: Configurable track lifecycle (max_age parameter)
- **Entry Time Recording**: Timestamp recording for wait time analysis
- **Position-Based Sorting**: Queue order determination via x-axis coordinate

### Logging & Monitoring
- **Comprehensive Logging**:
  - File-based logging for debugging and auditing
  - Console output for real-time monitoring
  - Error tracking with full stack traces
  
- **Structured Configuration**: Centralized config management for easy tuning

---

## 🏗️ Architecture

### Core Components

```
QueueDetectionSystem (Main Orchestrator)
├── PersonTracker (Multi-Object Tracking)
├── QueueAnalytics (Statistics Engine)
├── ResourceMonitor (System Monitoring)
└── Config (Configuration Management)
```

#### PersonTracker
- **IoU-Based Association**: Tracks persons across frames using Intersection over Union
- **Track Lifecycle**: Maintains tracks up to `max_age` frames
- **ID Persistence**: Unique identifier generation for consistent person tracking

#### QueueAnalytics
- **Windowed Statistics**: Maintains deque-based history for memory efficiency
- **Real-time Aggregation**: Mean, std, min, max calculations
- **Multi-metric Tracking**: Queue length, confidence, FPS, resources

#### ResourceMonitor
- **CPU Usage**: Per-frame CPU utilization percentage
- **Memory Usage**: Virtual memory consumption tracking
- **Timestamp Recording**: ISO format timestamps for correlation

---

## 🚀 Technical Highlights

### Type Safety & Code Quality
- ✅ Full type hints for all functions and class methods
- ✅ Dataclasses for structured data representation
- ✅ Comprehensive docstrings and documentation
- ✅ Exception handling with logging

### ML-Specific Features
- **Multi-Scale Detection**: Filters by bounding box height to reduce false positives
- **Confidence Visualization**: Color-coded detection boxes based on confidence:
  - 🟢 Green: High confidence (>0.8)
  - 🟠 Orange: Medium confidence (0.6-0.8)
  - 🔴 Red: Low confidence (<0.6)

### Embedded Systems Considerations
- **Memory Efficiency**: Fixed-size deques prevent unbounded memory growth
- **Frame Skipping**: Reduces computational load on resource-constrained devices
- **Configurable Thresholds**: Adapt to various hardware capabilities

### Production Readiness
- **Configuration Management**: Centralized `Config` dataclass for easy parameter tuning
- **Error Handling**: Graceful degradation with detailed logging
- **Data Export**: Multiple export formats (CSV, JSON) for downstream analysis
- **Structured Output**: Organized output directory for all generated files

---

## 📊 Output Files

The system generates three key output files in the `queue_output/` directory:

1. **queue_data.csv**
   - Real-time queue metrics per frame
   - Columns: timestamp, queue_length, avg_confidence, fps, cpu_percent, memory_percent

2. **queue_statistics.json**
   - Aggregate statistics summary
   - Includes: average queue length, max, min, std dev, total frames, runtime

3. **queue_detector.log**
   - Comprehensive execution log
   - Useful for debugging and performance analysis

---

## ⚙️ Configuration Parameters

```python
Config:
  QUEUE_Y_MIN = 200              # Top boundary of queue zone
  QUEUE_Y_MAX = 400              # Bottom boundary of queue zone
  CONF_THRESHOLD = 0.4           # Minimum detection confidence
  MIN_DETECTION_SIZE = 20        # Minimum person height in pixels
  MAX_DETECTION_SIZE = 800       # Maximum person height in pixels
  TRACKING_MAX_AGE = 30          # Frames to retain lost tracks
  IOU_THRESHOLD = 0.3            # IoU threshold for tracking association
  FRAME_SKIP = 1                 # Process every Nth frame
  ENABLE_RESOURCE_MONITORING = True
  ENABLE_CONFIDENCE_VISUALIZATION = True
```

---

## 🔧 Usage

### Basic Usage
```bash
python queuelenghth.py
```

### Exit
- Press `q` to gracefully shut down the system
- All data is automatically saved to `queue_output/`

### Customize Configuration
Edit the `Config` dataclass at the top of the script to adjust parameters:

```python
config = Config(
    QUEUE_Y_MIN=150,
    QUEUE_Y_MAX=450,
    CONF_THRESHOLD=0.5,
    FRAME_SKIP=2,  # Process every 2nd frame
)
```

---

## 📈 Performance Metrics

The system tracks and reports:
- **Detection Metrics**: Confidence scores, detection counts
- **Performance Metrics**: FPS, average FPS, total frames processed
- **Queue Metrics**: Length, average, max, min, standard deviation
- **Resource Metrics**: CPU%, Memory%, Timestamps

---

## 🎓 Learning & Interview Highlights

### ML/Computer Vision
- YOLOv8 integration and inference
- Multi-object tracking algorithms (IoU-based)
- Confidence-based filtering and validation
- Object detection metrics and analysis

### Software Engineering
- Object-oriented design with clear separation of concerns
- Type hints and dataclasses for maintainability
- Logging and error handling best practices
- Configuration management patterns

### Embedded Systems
- Resource monitoring (CPU, memory)
- Frame rate optimization techniques
- Efficient data structures (deques with maxlen)
- Performance profiling and analysis

### Data Science
- Statistical analysis (mean, std, percentiles)
- Real-time metrics calculation
- Data export in multiple formats (CSV, JSON)
- Historical data analysis

---

## 🔗 Dependencies

```
opencv-python>=4.8.0      # Computer vision library
numpy>=1.24.0             # Numerical computing
pandas>=2.0.0             # Data analysis
ultralytics>=8.0.0        # YOLOv8 implementation
psutil>=5.9.0             # System resource monitoring
```

---

## 🛠️ Future Enhancement Opportunities

1. **Multi-Zone Tracking**: Support multiple queue zones simultaneously
2. **Wait Time Estimation**: Predict wait times based on queue velocity
3. **Crowd Density Analysis**: Heat map generation for congestion areas
4. **Model Fine-tuning**: Train custom YOLO weights for specific scenarios
5. **Real-time Dashboard**: Web-based visualization using Streamlit/Dash
6. **Alert System**: Notifications when queue exceeds thresholds
7. **GPU Acceleration**: CUDA optimization for inference speed
8. **Async Processing**: Multi-threaded frame processing pipeline

---

## 📝 License

This project demonstrates advanced computer vision and software engineering practices for production-grade systems.

---

**Built for ML and Embedded Systems Excellence** 🚀
