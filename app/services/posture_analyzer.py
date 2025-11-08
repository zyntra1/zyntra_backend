"""
Posture Analysis Service using MediaPipe
Detects pose landmarks and classifies posture types
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import math
from datetime import datetime
import uuid


class PostureAnalyzer:
    """
    Analyzes video frames to detect human pose and classify posture
    Uses MediaPipe Pose for landmark detection
    """
    
    def __init__(self):
        """Initialize MediaPipe Pose"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Posture classification thresholds (in degrees)
        self.GOOD_NECK_ANGLE_MIN = 160
        self.FORWARD_HEAD_THRESHOLD = 145
        self.SLOUCH_SPINE_THRESHOLD = 150
        
    def analyze_video(
        self, 
        video_path: str, 
        fps_sample: int = 1,
        max_persons: int = 5
    ) -> Dict:
        """
        Analyze video and extract posture data for all detected persons
        
        Args:
            video_path: Path to video file
            fps_sample: Sample every Nth second (default: 1 = every second)
            max_persons: Maximum number of persons to track
            
        Returns:
            Dictionary with analysis results for each person
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Track persons across frames
        person_tracks = {}  # person_id -> list of posture snapshots
        session_id = str(uuid.uuid4())
        
        frame_idx = 0
        frames_processed = 0
        
        print(f"Analyzing video: {fps} FPS, {total_frames} frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on fps_sample (e.g., every 30 frames for 1 FPS sampling)
            if frame_idx % (fps * fps_sample) == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Extract posture data
                    posture_data = self._analyze_pose_landmarks(
                        results.pose_landmarks,
                        frame_idx,
                        session_id
                    )
                    
                    # Assign to person (simple approach: just use person_1, person_2, etc.)
                    # For demo, we'll track the first detected person
                    person_id = "person_1"
                    
                    if person_id not in person_tracks:
                        person_tracks[person_id] = []
                    
                    person_tracks[person_id].append(posture_data)
                    frames_processed += 1
            
            frame_idx += 1
        
        cap.release()
        
        return {
            'session_id': session_id,
            'video_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'frames_processed': frames_processed,
            'persons': person_tracks,
            'duration_seconds': total_frames / fps
        }
    
    def _analyze_pose_landmarks(
        self, 
        landmarks, 
        frame_number: int,
        session_id: str
    ) -> Dict:
        """
        Analyze pose landmarks and extract posture metrics
        
        Args:
            landmarks: MediaPipe pose landmarks
            frame_number: Current frame number
            session_id: Session identifier
            
        Returns:
            Dictionary with posture metrics
        """
        # Extract key landmarks
        keypoints = self._extract_keypoints(landmarks)
        
        # Calculate angles
        neck_angle = self._calculate_neck_angle(landmarks)
        shoulder_symmetry = self._calculate_shoulder_symmetry(landmarks)
        spine_score = self._calculate_spine_score(landmarks)
        
        # Classify posture
        posture_type, confidence = self._classify_posture(
            neck_angle, 
            shoulder_symmetry, 
            spine_score
        )
        
        # Detect activity
        activity = self._detect_activity(landmarks)
        
        return {
            'frame_number': frame_number,
            'session_id': session_id,
            'timestamp': datetime.utcnow(),
            'posture_type': posture_type,
            'confidence': confidence,
            'neck_angle': neck_angle,
            'shoulder_symmetry': shoulder_symmetry,
            'spine_score': spine_score,
            'activity': activity,
            'keypoints_json': keypoints
        }
    
    def _extract_keypoints(self, landmarks) -> Dict:
        """Extract all 33 keypoints as JSON"""
        keypoints = {}
        for idx, landmark in enumerate(landmarks.landmark):
            keypoints[f'point_{idx}'] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        return keypoints
    
    def _calculate_neck_angle(self, landmarks) -> float:
        """
        Calculate neck angle (forward head posture indicator)
        
        Angle between:
        - Shoulder center
        - Ear (approximated by nose)
        - Vertical line
        
        Returns angle in degrees (180 = perfectly straight)
        """
        try:
            # Get landmarks
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_ear = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            
            # Calculate shoulder center
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Use ear for head position
            head_x = left_ear.x
            head_y = left_ear.y
            
            # Calculate angle
            delta_x = head_x - shoulder_center_x
            delta_y = head_y - shoulder_center_y
            
            # Angle from vertical (negative y-axis)
            angle = math.degrees(math.atan2(abs(delta_x), abs(delta_y)))
            
            # Convert to 0-180 scale where 180 is straight
            neck_angle = 180 - abs(angle)
            
            return max(0, min(180, neck_angle))
        
        except Exception as e:
            print(f"Error calculating neck angle: {e}")
            return 160  # Default to good posture
    
    def _calculate_shoulder_symmetry(self, landmarks) -> float:
        """
        Calculate shoulder symmetry (slouch indicator)
        
        Returns:
            Float 0-1 where 1 = perfectly symmetric
        """
        try:
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate height difference
            height_diff = abs(left_shoulder.y - right_shoulder.y)
            
            # Convert to symmetry score (0-1)
            # Smaller difference = higher symmetry
            symmetry = max(0, 1 - (height_diff * 10))
            
            return symmetry
        
        except Exception as e:
            print(f"Error calculating shoulder symmetry: {e}")
            return 0.9  # Default to good symmetry
    
    def _calculate_spine_score(self, landmarks) -> float:
        """
        Calculate spine alignment score
        
        Returns:
            Float 0-100 where 100 = perfect alignment
        """
        try:
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Calculate centers
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            
            # Calculate spine angle from vertical
            delta_x = abs(shoulder_center_x - hip_center_x)
            delta_y = abs(shoulder_center_y - hip_center_y)
            
            if delta_y == 0:
                return 50
            
            angle = math.degrees(math.atan2(delta_x, delta_y))
            
            # Convert to score (0-100)
            # Perfect vertical = 100
            spine_score = max(0, 100 - (angle * 2))
            
            return spine_score
        
        except Exception as e:
            print(f"Error calculating spine score: {e}")
            return 70  # Default to decent score
    
    def _classify_posture(
        self, 
        neck_angle: float, 
        shoulder_symmetry: float, 
        spine_score: float
    ) -> Tuple[str, float]:
        """
        Classify posture type based on metrics
        
        Returns:
            Tuple of (posture_type, confidence)
        """
        confidence = 0.85  # Base confidence
        
        # Good posture
        if (neck_angle >= self.GOOD_NECK_ANGLE_MIN and 
            shoulder_symmetry >= 0.85 and 
            spine_score >= 80):
            return ("good", 0.95)
        
        # Forward head
        if neck_angle < self.FORWARD_HEAD_THRESHOLD:
            return ("forward_head", 0.90)
        
        # Slouched (low shoulder symmetry or poor spine alignment)
        if shoulder_symmetry < 0.7 or spine_score < 60:
            return ("slouched", 0.88)
        
        # Leaning (asymmetric shoulders but decent neck)
        if shoulder_symmetry < 0.8 and neck_angle >= self.FORWARD_HEAD_THRESHOLD:
            return ("leaning", 0.85)
        
        # Moderate posture (doesn't fit other categories well)
        return ("moderate", 0.75)
    
    def _detect_activity(self, landmarks) -> str:
        """
        Detect activity type (sitting, standing, walking)
        
        Simplified: Check knee angle and hip height
        """
        try:
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # Calculate knee angle
            hip_to_knee_y = left_knee.y - left_hip.y
            knee_to_ankle_y = left_ankle.y - left_knee.y
            
            # If knee is bent significantly, likely sitting
            if abs(hip_to_knee_y) < 0.3:
                return "sitting"
            else:
                return "standing"
        
        except Exception as e:
            return "unknown"
    
    def close(self):
        """Clean up resources"""
        self.pose.close()
