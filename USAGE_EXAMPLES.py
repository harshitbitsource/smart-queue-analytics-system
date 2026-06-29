#!/usr/bin/env python3
"""
Quick Start Guide and Usage Examples
Advanced Queue Detection System

This file demonstrates various use cases and configurations.
"""

# Example 1: Basic Usage with Default Configuration
# ===================================================
# Simply run the script:
#   python queuelenghth.py
# 
# This will:
# - Use default configuration
# - Start queue detection on webcam
# - Save results to queue_output/
# - Press 'q' to exit


# Example 2: Custom Configuration for Retail Store
# ==================================================
"""
from queuelenghth import QueueDetectionSystem, Config

config = Config(
    QUEUE_Y_MIN=150,           # Adjust for store layout
    QUEUE_Y_MAX=450,
    CONF_THRESHOLD=0.6,        # Higher confidence for retail
    MIN_DETECTION_SIZE=30,     # Filter small detections
    TRACKING_MAX_AGE=20,       # Faster resets
    ENABLE_RESOURCE_MONITORING=True,
)

system = QueueDetectionSystem(config)
system.run()
"""


# Example 3: Edge Device Deployment (Jetson Nano)
# ================================================
"""
from queuelenghth import QueueDetectionSystem, Config

config = Config(
    QUEUE_Y_MIN=200,
    QUEUE_Y_MAX=400,
    CONF_THRESHOLD=0.4,
    FRAME_SKIP=2,              # Process every 2nd frame
    TRACKING_MAX_AGE=30,
    ENABLE_RESOURCE_MONITORING=True,  # Monitor for thermal throttling
    ENABLE_CONFIDENCE_VISUALIZATION=True,
)

system = QueueDetectionSystem(config)
system.run()
"""


# Example 4: High-Performance Cloud Deployment
# ==============================================
"""
from queuelenghth import QueueDetectionSystem, Config

config = Config(
    QUEUE_Y_MIN=200,
    QUEUE_Y_MAX=400,
    CONF_THRESHOLD=0.8,        # Very strict detection
    FRAME_SKIP=1,              # Process every frame
    TRACKING_MAX_AGE=50,       # Longer track history
    ENABLE_RESOURCE_MONITORING=False,  # Resources not constrained
    MIN_DETECTION_SIZE=25,
    MAX_DETECTION_SIZE=500,
)

system = QueueDetectionSystem(config)
system.run()
"""


# Example 5: Multi-Camera System (Production Use)
# ================================================
"""
from queuelenghth import QueueDetectionSystem, Config
import threading
import os

def monitor_queue(camera_id, zone_name):
    '''Monitor a single camera'''
    config = Config(
        QUEUE_Y_MIN=200,
        QUEUE_Y_MAX=400,
        CONF_THRESHOLD=0.5,
        OUTPUT_DIR=f'queue_output_{zone_name}',
    )
    
    # Create custom system with specific camera
    system = QueueDetectionSystem(config)
    # Note: Modify QueueDetectionSystem to accept camera_id parameter
    system.run()

# Launch threads for multiple cameras
threads = []
zones = ['entrance', 'checkout', 'customer_service']

for i, zone in enumerate(zones):
    t = threading.Thread(target=monitor_queue, args=(i, zone))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
"""


# Example 6: Analyzing Output Data
# ==================================
"""
import pandas as pd
import json

# Load queue statistics
with open('queue_output/queue_statistics.json', 'r') as f:
    stats = json.load(f)

# Load time-series data
df = pd.read_csv('queue_output/queue_data.csv')

# Analyze peak times
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S')
peak_hour = df.groupby(df['timestamp'].dt.hour)['queue_length'].mean()
print("Average queue length by hour:")
print(peak_hour)

# Detect anomalies
mean_queue = df['queue_length'].mean()
std_queue = df['queue_length'].std()
anomalies = df[df['queue_length'] > mean_queue + 2*std_queue]
print(f"\nAnomalies detected: {len(anomalies)} frames")

# Performance analysis
print(f"\nAverage FPS: {df['fps'].mean():.1f}")
print(f"Resource Usage - CPU: {df['cpu_percent'].mean():.1f}%, Memory: {df['memory_percent'].mean():.1f}%")
"""


# Example 7: Integration with Flask Web Server
# =============================================
"""
from flask import Flask, jsonify
from queuelenghth import QueueDetectionSystem, Config
import threading
import json

app = Flask(__name__)
system = None
latest_stats = {}

def run_detection():
    '''Run detection in background thread'''
    global system, latest_stats
    config = Config()
    system = QueueDetectionSystem(config)
    system.run()

@app.route('/api/queue-stats', methods=['GET'])
def get_stats():
    '''Get current queue statistics'''
    if system:
        stats = system.analytics.get_statistics()
        return jsonify(stats)
    return jsonify({"error": "System not initialized"}), 500

@app.route('/api/queue-data', methods=['GET'])
def get_data():
    '''Get queue data as JSON'''
    if system:
        data = [d for d in system.queue_data]
        return jsonify(data)
    return jsonify({"error": "System not initialized"}), 500

# Start detection in background
thread = threading.Thread(target=run_detection, daemon=True)
thread.start()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
"""


# Example 8: Docker Deployment
# =============================
"""
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY queuelenghth.py .
COPY config_template.json .

# For webcam access: docker run --device /dev/video0 queue-detector

ENV PYTHONUNBUFFERED=1
CMD ["python", "queuelenghth.py"]
"""


# Example 9: Performance Profiling
# =================================
"""
import cProfile
import pstats
from queuelenghth import QueueDetectionSystem, Config

config = Config(FRAME_SKIP=1)
system = QueueDetectionSystem(config)

# Profile the execution
profiler = cProfile.Profile()
profiler.enable()

# Run for limited frames
for _ in range(100):
    ret, frame = system.cap.read()
    if not ret:
        break
    detections, _ = system.process_frame(frame)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
"""


# Example 10: Continuous Monitoring with Logging
# ================================================
"""
import logging
from queuelenghth import setup_logging, QueueDetectionSystem, Config

# Set up detailed logging
logger = setup_logging('queue_detector_production.log')

try:
    config = Config(
        ENABLE_RESOURCE_MONITORING=True,
        FRAME_SKIP=1,
    )
    
    system = QueueDetectionSystem(config)
    logger.info("Queue detection system started")
    system.run()
    
except Exception as e:
    logger.error(f"Critical error: {e}", exc_info=True)
finally:
    logger.info("System shutdown complete")
"""


if __name__ == "__main__":
    print(__doc__)
    print("\nRefer to the examples above for different use cases.")
    print("Copy and paste relevant examples into your code as needed.")
