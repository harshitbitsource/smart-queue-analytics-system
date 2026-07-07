"""
Advanced Queue Length Counter with ML Analytics
Features:
- Real-time person detection and tracking
- Multi-zone queue analysis
- Advanced statistics and analytics
- Performance monitoring
- Resource utilization tracking
- Confidence-based filtering
- Historical data analysis
"""

import os
import cv2
import numpy as np
import pandas as pd
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import psutil
import json
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    """Configuration parameters for queue detection system"""
    QUEUE_Y_MIN: int = 200
    QUEUE_Y_MAX: int = 400
    CONF_THRESHOLD: float = 0.4
    MIN_DETECTION_SIZE: int = 20  # Minimum bounding box height
    MAX_DETECTION_SIZE: int = 800  # Maximum bounding box height
    TRACKING_MAX_AGE: int = 30  # Frames to keep lost tracks
    IOU_THRESHOLD: float = 0.3  # IoU threshold for tracking
    FRAME_SKIP: int = 1  # Process every Nth frame for optimization
    ENABLE_RESOURCE_MONITORING: bool = True
    ENABLE_CONFIDENCE_VISUALIZATION: bool = True
    OUTPUT_DIR: str = "queue_output"
    MODEL_NAME: str = "yolov8n.pt"
    USE_YOLO: bool = True

config = Config()

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(log_file: str = "queue_detector.log") -> logging.Logger:
    """Configure logging with file and console handlers"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler = logging.FileHandler(f"{config.OUTPUT_DIR}/{log_file}")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ============================================================================
# PERSON TRACKING CLASS
# ============================================================================
@dataclass
class Detection:
    """Data class for a detected person"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int = -1
    entry_time: float = 0.0
    
    @property
    def center(self) -> Tuple[int, int]:
        """Calculate bounding box center"""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def area(self) -> int:
        """Calculate bounding box area"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    @property
    def height(self) -> int:
        """Calculate bounding box height"""
        return self.y2 - self.y1


class PersonTracker:
    """Multi-object tracker with IoU-based association"""
    
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Tuple[Detection, int]] = {}  # id -> (detection, age)
        self.next_id = 0

    @staticmethod
    def iou(box1: Detection, box2: Detection) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1_inter = max(box1.x1, box2.x1)
        y1_inter = max(box1.y1, box2.y1)
        x2_inter = min(box1.x2, box2.x2)
        y2_inter = min(box1.y2, box2.y2)

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        box1_area = box1.area
        box2_area = box2.area
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update tracks with new detections"""
        matched_detections = [False] * len(detections)

        for track_id, (track_det, age) in list(self.tracks.items()):
            best_match_idx = -1
            best_iou = self.iou_threshold

            for det_idx, det in enumerate(detections):
                if matched_detections[det_idx]:
                    continue

                current_iou = self.iou(track_det, det)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_match_idx = det_idx

            if best_match_idx >= 0:
                detections[best_match_idx].track_id = track_id
                self.tracks[track_id] = (detections[best_match_idx], 0)
                matched_detections[best_match_idx] = True
            else:
                self.tracks[track_id] = (track_det, age + 1)

        self.tracks = {
            tid: (det, age)
            for tid, (det, age) in self.tracks.items()
            if age < self.max_age
        }

        for det_idx, det in enumerate(detections):
            if not matched_detections[det_idx]:
                det.track_id = self.next_id
                det.entry_time = time.time()
                self.tracks[self.next_id] = (det, 0)
                self.next_id += 1

        return detections


class FallbackPersonDetector:
    """Fallback detector using OpenCV HOG person detector."""

    def __init__(self):
        if not hasattr(cv2, "HOGDescriptor") or not hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector"):
            raise RuntimeError(
                "OpenCV HOGDescriptor is unavailable in this build. "
                "Install ultralytics for YOLO or use a different OpenCV package."
            )
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[Detection]:
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        detections = []
        for (x, y, w, h), weight in zip(rects, weights):
            detections.append(Detection(x, y, x + w, y + h, float(weight)))
        return detections


# ============================================================================
# ANALYTICS ENGINE
# ============================================================================
class QueueAnalytics:
    """Real-time and historical analytics for queue data"""
    
    def __init__(self, window_size: int = 100):
        self.queue_history: deque = deque(maxlen=window_size)
        self.confidence_history: deque = deque(maxlen=window_size)
        self.fps_history: deque = deque(maxlen=window_size)
        self.resource_history: deque = deque(maxlen=window_size)
        self.start_time = time.time()
        self.frame_count = 0
    
    def add_frame_data(
        self, 
        queue_length: int, 
        confidences: List[float], 
        fps: float,
        cpu_percent: float,
        memory_percent: float
    ):
        """Add data from current frame"""
        self.queue_history.append(queue_length)
        avg_conf = np.mean(confidences) if confidences else 0
        self.confidence_history.append(avg_conf)
        self.fps_history.append(fps)
        self.resource_history.append({
            'cpu': cpu_percent,
            'memory': memory_percent,
            'timestamp': time.time()
        })
        self.frame_count += 1
    
    def get_statistics(self) -> Dict:
        """Calculate comprehensive statistics"""
        if not self.queue_history:
            return {}
        
        queue_data = list(self.queue_history)
        conf_data = list(self.confidence_history)
        fps_data = list(self.fps_history)
        
        return {
            'current_queue_length': queue_data[-1],
            'avg_queue_length': np.mean(queue_data),
            'max_queue_length': np.max(queue_data),
            'min_queue_length': np.min(queue_data),
            'std_queue_length': np.std(queue_data),
            'avg_confidence': np.mean(conf_data) if conf_data else 0,
            'current_fps': fps_data[-1] if fps_data else 0,
            'avg_fps': np.mean(fps_data) if fps_data else 0,
            'total_frames': self.frame_count,
            'runtime_seconds': time.time() - self.start_time,
        }


# ============================================================================
# RESOURCE MONITOR
# ============================================================================
class ResourceMonitor:
    """Monitor system resource utilization"""
    
    @staticmethod
    def get_system_stats() -> Dict:
        """Get current CPU and memory usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'timestamp': datetime.now().isoformat()
        }


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class QueueDetectionSystem:
    """Main system for queue detection and analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)

        logger.info("Initializing Queue Detection System")
        self.model = None
        self.detector = None

        if self.config.USE_YOLO and YOLO_AVAILABLE:
            logger.info(f"Loading YOLO model: {config.MODEL_NAME}")
            try:
                self.model = YOLO(config.MODEL_NAME)
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.warning(
                    f"YOLO load failed: {e}. Falling back to OpenCV HOG detector."
                )
                self.model = None
                self.detector = FallbackPersonDetector()
        else:
            if self.config.USE_YOLO and not YOLO_AVAILABLE:
                logger.warning(
                    "Ultralytics YOLO is not installed; using OpenCV HOG fallback detector."
                )
            self.detector = FallbackPersonDetector()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            raise RuntimeError("Cannot access camera")

        self.tracker = PersonTracker(
            max_age=config.TRACKING_MAX_AGE,
            iou_threshold=config.IOU_THRESHOLD
        )
        self.analytics = QueueAnalytics()
        self.queue_data = []
        self.prev_time = time.time()
        self.frame_index = 0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Detection], Optional[List]]:
        """Detect and track persons in frame"""
        detections: List[Detection] = []
        results = None

        if self.model is not None:
            results = self.model(frame)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Detect only people (class 0) with good confidence
                    if cls == 0 and conf > self.config.CONF_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        height = y2 - y1

                        # Filter by size
                        if (
                            self.config.MIN_DETECTION_SIZE < height <
                            self.config.MAX_DETECTION_SIZE
                        ):
                            cy = (y1 + y2) // 2

                            # Check if in queue zone
                            if self.config.QUEUE_Y_MIN < cy < self.config.QUEUE_Y_MAX:
                                detections.append(
                                    Detection(x1, y1, x2, y2, conf)
                                )
        else:
            if self.detector is not None:
                detections = self.detector.detect(frame)
                filtered: List[Detection] = []
                for det in detections:
                    if (
                        self.config.MIN_DETECTION_SIZE < det.height <
                        self.config.MAX_DETECTION_SIZE
                    ):
                        cy = (det.y1 + det.y2) // 2
                        if self.config.QUEUE_Y_MIN < cy < self.config.QUEUE_Y_MAX:
                            filtered.append(det)
                detections = filtered

        # Sort by X-axis (queue order)
        detections.sort(key=lambda d: d.center[0])

        # Update tracks
        tracked_detections = self.tracker.update(detections)

        return tracked_detections, results
    
    def draw_visualization(
        self, 
        frame: np.ndarray, 
        detections: List[Detection],
        fps: float,
        stats: Dict
    ) -> np.ndarray:
        """Draw annotations and UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Draw queue region
        cv2.rectangle(
            frame,
            (0, self.config.QUEUE_Y_MIN),
            (w, self.config.QUEUE_Y_MAX),
            (255, 0, 0),
            2
        )
        cv2.putText(
            frame,
            "Queue Zone",
            (10, self.config.QUEUE_Y_MIN - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )
        
        # Draw detections with tracking IDs
        for i, person in enumerate(detections):
            color = self._get_color_by_confidence(person.confidence)
            
            # Draw bounding box
            cv2.rectangle(
                frame,
                (person.x1, person.y1),
                (person.x2, person.y2),
                color,
                2
            )
            
            # Draw position and ID
            label = f"ID:{person.track_id} Pos:{i+1}"
            cv2.putText(
                frame,
                label,
                (person.x1, person.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            
            # Draw confidence if enabled
            if self.config.ENABLE_CONFIDENCE_VISUALIZATION:
                conf_text = f"Conf: {person.confidence:.2f}"
                cv2.putText(
                    frame,
                    conf_text,
                    (person.x1, person.y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
        
        # Draw statistics panel
        self._draw_stats_panel(frame, detections, fps, stats)
        
        return frame
    
    @staticmethod
    def _get_color_by_confidence(confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence level"""
        if confidence > 0.8:
            return (0, 255, 0)  # Green - high confidence
        elif confidence > 0.6:
            return (0, 165, 255)  # Orange - medium confidence
        else:
            return (0, 0, 255)  # Red - low confidence
    
    def _draw_stats_panel(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float,
        stats: Dict
    ):
        """Draw statistics panel on frame"""
        panel_height = 180
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        y_offset = 25
        line_height = 20
        
        stats_text = [
            f"Queue Length: {len(detections)}",
            f"FPS: {fps:.1f}",
            f"Avg Confidence: {stats.get('avg_confidence', 0):.2f}",
            f"Avg Queue: {stats.get('avg_queue_length', 0):.1f}",
            f"Max Queue: {stats.get('max_queue_length', 0)}",
            f"Frames: {stats.get('total_frames', 0)}",
        ]
        
        if self.config.ENABLE_RESOURCE_MONITORING and self.analytics.resource_history:
            res = self.analytics.resource_history[-1]
            stats_text.append(f"CPU: {res['cpu']:.1f}% | MEM: {res['memory']:.1f}%")
        
        for i, text in enumerate(stats_text):
            cv2.putText(
                frame,
                text,
                (10, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1
            )
    
    def run(self):
        """Main execution loop"""
        logger.info("Starting queue detection")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Skip frames for optimization
                if self.frame_index % self.config.FRAME_SKIP != 0:
                    self.frame_index += 1
                    continue
                
                # Process frame
                detections, results = self.process_frame(frame)
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
                self.prev_time = curr_time
                
                # Get resource stats
                resource_stats = ResourceMonitor.get_system_stats() if \
                    self.config.ENABLE_RESOURCE_MONITORING else \
                    {'cpu_percent': 0, 'memory_percent': 0}
                
                # Update analytics
                confidences = [d.confidence for d in detections]
                self.analytics.add_frame_data(
                    len(detections),
                    confidences,
                    fps,
                    resource_stats['cpu_percent'],
                    resource_stats['memory_percent']
                )
                
                # Get statistics
                stats = self.analytics.get_statistics()
                
                # Draw visualization
                frame = self.draw_visualization(frame, detections, fps, stats)
                
                # Save data
                timestamp = time.strftime("%H:%M:%S")
                self.queue_data.append({
                    'timestamp': timestamp,
                    'queue_length': len(detections),
                    'avg_confidence': stats.get('avg_confidence', 0),
                    'fps': fps,
                    'cpu_percent': resource_stats['cpu_percent'],
                    'memory_percent': resource_stats['memory_percent']
                })
                
                # Display
                cv2.imshow("Advanced Queue Detection System", frame)
                
                # Check for exit
                key = cv2.waitKey(1)
                if key == ord('q'):
                    logger.info("Exit signal received")
                    break
                
                self.frame_index += 1
        
        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Cleanup and save data"""
        logger.info("Shutting down system")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Save queue data
        df = pd.DataFrame(self.queue_data)
        output_file = f"{self.config.OUTPUT_DIR}/queue_data.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Queue data saved to {output_file}")


def main() -> int:
    """Run the queue detection system."""
    try:
        system = QueueDetectionSystem(config)
        system.run()
    except Exception as e:
        logger.error(f"Application terminated with error: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())