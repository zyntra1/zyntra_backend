"""
Gait Recognition Service using OpenGait
Handles person detection, gait feature extraction, and matching
"""
import os
import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import json

# Hardcoded YOLO model path
YOLO_MODEL_PATH = "yolov8n.pt"


class GaitProcessor:
    """
    Main class for gait recognition processing
    Integrates YOLO for person detection and OpenGait for gait recognition
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the gait processor
        
        Args:
            model_path: Path to OpenGait model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize YOLO for person detection
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # OpenGait model will be loaded when model file is available
        self.gait_model = None
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self._load_opengait_model(model_path)
    
    def _load_opengait_model(self, model_path: str):
        """
        Load OpenGait model (GaitBase)
        Note: This is a placeholder - actual implementation depends on OpenGait structure
        """
        try:
            # Load the model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"OpenGait model loaded from {model_path}")
            
            # This will need to be adjusted based on actual OpenGait model structure
            # For now, storing checkpoint for later use
            self.gait_model = checkpoint
            
        except Exception as e:
            print(f"Error loading OpenGait model: {e}")
            print("Please ensure the GaitBase model file is in the correct location")
    
    def detect_persons(self, video_path: str, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect persons in video using YOLO
        
        Args:
            video_path: Path to video file
            conf_threshold: Confidence threshold for detection
            
        Returns:
            List of detections with frame-wise bounding boxes
        """
        cap = cv2.VideoCapture(video_path)
        detections = []
        frame_idx = 0
        
        # Track persons across frames (simple tracking by IoU)
        person_tracks = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = self.yolo_model(frame, conf=conf_threshold, classes=[0])  # class 0 is person
            
            # Process detections for this frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Assign to track or create new track
                    bbox = {
                        'frame': frame_idx,
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'conf': float(conf)
                    }
                    
                    # Simple tracking: find overlapping track
                    matched = False
                    for track in person_tracks:
                        if self._iou(bbox, track['boxes'][-1]) > 0.3:
                            track['boxes'].append(bbox)
                            matched = True
                            break
                    
                    if not matched:
                        person_tracks.append({
                            'track_id': len(person_tracks),
                            'boxes': [bbox]
                        })
            
            frame_idx += 1
        
        cap.release()
        
        # Filter tracks with sufficient frames (at least 10 frames)
        valid_tracks = [track for track in person_tracks if len(track['boxes']) >= 10]
        
        return valid_tracks
    
    def _iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def extract_gait_features(self, video_path: str, person_track: Dict) -> Optional[np.ndarray]:
        """
        Extract gait features for a person track
        
        Args:
            video_path: Path to video file
            person_track: Dictionary containing person's bounding boxes across frames
            
        Returns:
            Gait embedding vector (numpy array)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Extract person crops from video
            person_crops = []
            for bbox in person_track['boxes']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, bbox['frame'])
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Crop person region
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # Resize to standard size (64x64 for simplicity)
                    crop = cv2.resize(crop, (64, 64))
                    person_crops.append(crop)
            
            cap.release()
            
            if len(person_crops) == 0:
                return None
            
            # For demo purposes without actual OpenGait model:
            # Create a simple feature by averaging appearance features
            # In production, this should use the actual GaitBase model
            
            if self.gait_model is None:
                # Fallback: Use simple appearance features
                features = self._extract_simple_features(person_crops)
            else:
                # Use actual OpenGait model
                features = self._extract_opengait_features(person_crops)
            
            return features
            
        except Exception as e:
            print(f"Error extracting gait features: {e}")
            return None
    
    def _extract_simple_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract simple appearance-based features (fallback method)
        Creates a 512-dimensional feature vector
        """
        # Convert crops to grayscale and extract features
        features = []
        for crop in crops:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Resize to fixed size for consistency
            gray = cv2.resize(gray, (32, 64))
            features.append(gray.flatten())
        
        # Average features across frames
        features_array = np.array(features)
        avg_features = np.mean(features_array, axis=0)
        
        # Normalize
        avg_features = avg_features / (np.linalg.norm(avg_features) + 1e-8)
        
        # Pad or truncate to 512 dimensions
        if len(avg_features) < 512:
            avg_features = np.pad(avg_features, (0, 512 - len(avg_features)))
        else:
            avg_features = avg_features[:512]
        
        return avg_features.astype(np.float32)
    
    def _extract_opengait_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features using OpenGait model
        TODO: Implement actual OpenGait inference
        """
        # This is a placeholder for actual OpenGait implementation
        # When the model is available, implement proper preprocessing and inference
        
        # For now, fall back to simple features
        return self._extract_simple_features(crops)
    
    def match_gait_embedding(self, query_embedding: np.ndarray, 
                            database_embeddings: List[Tuple[int, np.ndarray]],
                            threshold: float = 0.6) -> Tuple[Optional[int], float]:
        """
        Match query embedding against database
        
        Args:
            query_embedding: Gait embedding to match
            database_embeddings: List of (user_id, embedding) tuples
            threshold: Similarity threshold for matching
            
        Returns:
            (matched_user_id, confidence_score) or (None, 0) if no match
        """
        if len(database_embeddings) == 0:
            return None, 0.0
        
        # Calculate cosine similarity with all database embeddings
        best_match_id = None
        best_score = 0.0
        
        for user_id, db_embedding in database_embeddings:
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                db_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_match_id = user_id
        
        # Only return match if above threshold
        if best_score >= threshold:
            return best_match_id, float(best_score)
        else:
            return None, float(best_score)
    
    def create_annotated_video(self, input_video_path: str, output_video_path: str,
                              detections: List[Dict], recognition_results: List[Dict]):
        """
        Create annotated video with bounding boxes and recognition labels
        
        Args:
            input_video_path: Original video path
            output_video_path: Path to save annotated video
            detections: List of person detections
            recognition_results: List of recognition results with user info
        """
        cap = cv2.VideoCapture(input_video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw bounding boxes for all detections in this frame
            for det_idx, detection in enumerate(detections):
                for bbox in detection['boxes']:
                    if bbox['frame'] == frame_idx:
                        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                        
                        # Get recognition result for this detection
                        result = recognition_results[det_idx] if det_idx < len(recognition_results) else None
                        
                        # Choose color based on recognition status
                        if result and result.get('is_recognized'):
                            color = (0, 255, 0)  # Green for recognized
                            label = f"{result.get('username', 'Unknown')} ({result.get('confidence', 0):.2f})"
                        else:
                            color = (0, 0, 255)  # Red for unknown
                            label = "Unknown"
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label background
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        print(f"Annotated video saved to {output_video_path}")


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize numpy array to bytes for database storage"""
    return pickle.dumps(embedding)


def deserialize_embedding(data: bytes) -> np.ndarray:
    """Deserialize bytes to numpy array"""
    return pickle.loads(data)
