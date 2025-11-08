"""
Posture Analysis Service using MediaPipe and YOLO
Detects pose landmarks and classifies posture types for multiple persons
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import math
from datetime import datetime
import uuid
from collections import defaultdict
from ultralytics import YOLO
import os


class PostureAnalyzer:
    """
    Analyzes video frames to detect human pose and classify posture
    Uses MediaPipe Pose for landmark detection
    """
    
    def __init__(self):
        """Initialize YOLO for person detection and MediaPipe for pose analysis"""
        # YOLO for person detection
        model_path = "yolov8n.pt"
        if not os.path.exists(model_path):
            print(f"YOLO model not found at {model_path}, downloading...")
        self.yolo_model = YOLO(model_path)
        
        # MediaPipe for pose landmark detection
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Posture classification thresholds (in degrees)
        self.GOOD_NECK_ANGLE_MIN = 160
        self.FORWARD_HEAD_THRESHOLD = 145
        self.SLOUCH_SPINE_THRESHOLD = 150
        
        # Person tracking parameters
        self.PERSON_DISTANCE_THRESHOLD = 0.3  # Max distance to consider same person
        self.active_persons = {}  # person_id -> last known bbox center
        
    def analyze_video(
        self, 
        video_path: str, 
        fps_sample: Optional[int] = 1,
        max_persons: int = 5
    ) -> Dict:
        """
        Analyze video and extract posture data for all detected persons
        Uses YOLO for person detection and MediaPipe for pose analysis
        
        Args:
            video_path: Path to video file
            fps_sample: Sample every Nth second (default: 1 = every second, None = process all frames)
            max_persons: Maximum number of persons to track
            
        Returns:
            Dictionary with analysis results for each person
        """
        # Create a fresh MediaPipe Pose instance for this video
        pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Track persons across frames
            person_tracks = defaultdict(list)  # person_id -> list of posture snapshots
            session_id = str(uuid.uuid4())
            self.active_persons = {}  # Reset for new video
            next_person_id = 1
            
            frame_idx = 0
            frames_processed = 0
            frames_with_detection = 0
            total_persons_detected_in_frame = 0
            frames_analyzed = 0  # Frames actually analyzed (not skipped)
            
            if fps_sample is None:
                print(f"Analyzing video: {fps} FPS, {total_frames} frames - Processing ALL frames")
                print(f"Expected analysis time: This may take a while for long videos...")
            else:
                print(f"Analyzing video: {fps} FPS, {total_frames} frames - Sampling every {fps_sample} second(s)")
            
            # Progress tracking
            last_progress_print = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Print progress every 10%
                progress_pct = (frame_idx / total_frames) * 100
                if progress_pct - last_progress_print >= 10:
                    print(f"Progress: {progress_pct:.0f}% ({frame_idx}/{total_frames} frames)")
                    last_progress_print = progress_pct
                
                # Determine if we should process this frame
                should_process = False
                if fps_sample is None:
                    # Process all frames
                    should_process = True
                else:
                    # Sample frames based on fps_sample
                    should_process = (frame_idx % (fps * fps_sample) == 0)
                
                if should_process:
                    frames_analyzed += 1
                    
                    # Detect persons using YOLO
                    results = self.yolo_model(frame, verbose=False)
                    
                    persons_in_frame = []
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Class 0 is 'person' in COCO dataset
                            if int(box.cls[0]) == 0:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = float(box.conf[0])
                                
                                if confidence > 0.5:  # Only consider confident detections
                                    persons_in_frame.append({
                                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                        'confidence': confidence
                                    })
                    
                    # Log detections for this frame
                    if persons_in_frame:
                        confidences = [f"{p['confidence']:.2f}" for p in persons_in_frame]
                        print(f"Frame {frame_idx}: Detected {len(persons_in_frame)} person(s) - confidences: {confidences}")
                    
                    if persons_in_frame:
                        frames_with_detection += 1
                        total_persons_detected_in_frame += len(persons_in_frame)
                        
                        # Process each detected person
                        for idx, person_bbox_data in enumerate(persons_in_frame):
                            bbox = person_bbox_data['bbox']
                            x1, y1, x2, y2 = bbox
                            
                            # Crop person from frame
                            person_crop = frame[y1:y2, x1:x2]
                            
                            if person_crop.size == 0:
                                print(f"  Person {idx+1}: Skipped - invalid crop")
                                continue
                            
                            # Convert to RGB for MediaPipe
                            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                            
                            # Analyze pose with MediaPipe
                            pose_results = pose.process(rgb_crop)
                            
                            if pose_results.pose_landmarks:
                                # Calculate person center from bbox
                                bbox_center = ((x1 + x2) / 2 / frame.shape[1], 
                                              (y1 + y2) / 2 / frame.shape[0])
                                
                                # Match to existing person or create new one
                                person_id = self._match_or_create_person(
                                    bbox_center, 
                                    next_person_id, 
                                    max_persons
                                )
                                
                                if person_id:
                                    # Update next_person_id if we created a new person
                                    if isinstance(person_id, int) and person_id >= next_person_id:
                                        next_person_id = person_id + 1
                                    
                                    # Convert to string format
                                    person_key = f"person_{person_id}"
                                    
                                    # Extract posture data
                                    posture_data = self._analyze_pose_landmarks(
                                        pose_results.pose_landmarks,
                                        frame_idx,
                                        session_id
                                    )
                                    
                                    # Add bbox info to posture data
                                    posture_data['bbox'] = bbox
                                    posture_data['detection_confidence'] = person_bbox_data['confidence']
                                    
                                    person_tracks[person_key].append(posture_data)
                                    frames_processed += 1
                                    
                                    print(f"  Person {idx+1} → {person_key}: Posture={posture_data['posture_type']}, Activity={posture_data['activity']}")
                                else:
                                    print(f"  Person {idx+1}: Skipped - max persons limit reached")
                            else:
                                print(f"  Person {idx+1}: No pose landmarks detected")
                
                frame_idx += 1
            
            cap.release()
            
            # Filter out persons with too few detections (likely noise)
            min_detections = 3
            filtered_tracks = {
                person_id: snapshots 
                for person_id, snapshots in person_tracks.items() 
                if len(snapshots) >= min_detections
            }
            
            avg_persons_per_frame = total_persons_detected_in_frame / frames_with_detection if frames_with_detection > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"{'ANALYSIS COMPLETE':^60}")
            print(f"{'='*60}")
            print(f"Video Statistics:")
            print(f"  - Total frames in video: {total_frames}")
            print(f"  - Video duration: {total_frames / fps:.2f} seconds")
            print(f"  - FPS: {fps}")
            print(f"\nProcessing Statistics:")
            print(f"  - Frames analyzed (sent to YOLO): {frames_analyzed}")
            print(f"  - Frames with person detections: {frames_with_detection}")
            print(f"  - Successful pose analyses: {frames_processed}")
            print(f"  - Average persons per frame: {avg_persons_per_frame:.2f}")
            print(f"\nPerson Tracking:")
            print(f"  - Unique persons detected: {len(person_tracks)}")
            print(f"  - Persons after filtering (min {min_detections} detections): {len(filtered_tracks)}")
            
            for person_id, snapshots in filtered_tracks.items():
                print(f"\n  {person_id}:")
                print(f"    - Total snapshots: {len(snapshots)}")
                
                # Calculate posture distribution
                postures = [s['posture_type'] for s in snapshots]
                from collections import Counter
                posture_counts = Counter(postures)
                print(f"    - Postures: {dict(posture_counts)}")
                
                # Calculate activity distribution
                activities = [s['activity'] for s in snapshots]
                activity_counts = Counter(activities)
                print(f"    - Activities: {dict(activity_counts)}")
                
                # Average metrics
                avg_neck = sum(s['neck_angle'] for s in snapshots) / len(snapshots)
                avg_spine = sum(s['spine_score'] for s in snapshots) / len(snapshots)
                print(f"    - Avg neck angle: {avg_neck:.1f}°")
                print(f"    - Avg spine score: {avg_spine:.1f}")
            
            print(f"{'='*60}\n")
            
            return {
                'session_id': session_id,
                'video_path': video_path,
                'fps': fps,
                'total_frames': total_frames,
                'frames_analyzed': frames_analyzed,
                'frames_processed': frames_processed,
                'frames_with_detection': frames_with_detection,
                'persons': filtered_tracks,
                'duration_seconds': total_frames / fps,
                'persons_detected': len(filtered_tracks),
                'avg_persons_per_frame': round(avg_persons_per_frame, 2)
            }
        
        finally:
            # Clean up MediaPipe resources
            pose.close()
    
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
        Uses multiple indicators for accurate classification
        """
        try:
            # Get key landmarks
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate hip center
            hip_center_y = (left_hip.y + right_hip.y) / 2
            
            # Calculate knee center
            knee_center_y = (left_knee.y + right_knee.y) / 2
            
            # Calculate ankle center
            ankle_center_y = (left_ankle.y + right_ankle.y) / 2
            
            # Calculate shoulder center
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Calculate torso length (shoulder to hip)
            torso_length = abs(hip_center_y - shoulder_center_y)
            
            # Calculate upper leg length (hip to knee)
            upper_leg_length = abs(knee_center_y - hip_center_y)
            
            # Calculate lower leg length (knee to ankle)
            lower_leg_length = abs(ankle_center_y - knee_center_y)
            
            # Calculate knee angle using vectors
            left_knee_angle = self._calculate_angle_3_points(
                (left_hip.x, left_hip.y),
                (left_knee.x, left_knee.y),
                (left_ankle.x, left_ankle.y)
            )
            
            right_knee_angle = self._calculate_angle_3_points(
                (right_hip.x, right_hip.y),
                (right_knee.x, right_knee.y),
                (right_ankle.x, right_ankle.y)
            )
            
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            
            # Sitting detection criteria:
            # 1. Knee angle is acute (< 120 degrees) - bent legs
            # 2. Upper leg is more horizontal than vertical (small y-distance)
            # 3. Hip is relatively high in frame (not standing fully extended)
            
            if avg_knee_angle < 120 and upper_leg_length < torso_length * 0.8:
                return "sitting"
            
            # Standing detection criteria:
            # 1. Knee angle is more straight (> 140 degrees)
            # 2. Legs are relatively extended
            elif avg_knee_angle > 140:
                return "standing"
            
            # In-between or transitioning
            else:
                # Use upper leg length as tiebreaker
                if upper_leg_length < torso_length * 0.9:
                    return "sitting"
                else:
                    return "standing"
        
        except Exception as e:
            print(f"Error detecting activity: {e}")
            return "unknown"
    
    def _calculate_angle_3_points(
        self, 
        point_a: Tuple[float, float], 
        point_b: Tuple[float, float], 
        point_c: Tuple[float, float]
    ) -> float:
        """
        Calculate angle at point_b formed by point_a, point_b, point_c
        
        Returns angle in degrees
        """
        try:
            # Vectors
            ba_x = point_a[0] - point_b[0]
            ba_y = point_a[1] - point_b[1]
            bc_x = point_c[0] - point_b[0]
            bc_y = point_c[1] - point_b[1]
            
            # Dot product
            dot_product = ba_x * bc_x + ba_y * bc_y
            
            # Magnitudes
            magnitude_ba = math.sqrt(ba_x**2 + ba_y**2)
            magnitude_bc = math.sqrt(bc_x**2 + bc_y**2)
            
            # Avoid division by zero
            if magnitude_ba == 0 or magnitude_bc == 0:
                return 180.0
            
            # Cosine of angle
            cos_angle = dot_product / (magnitude_ba * magnitude_bc)
            
            # Clamp to valid range for arccos
            cos_angle = max(-1.0, min(1.0, cos_angle))
            
            # Angle in degrees
            angle = math.degrees(math.acos(cos_angle))
            
            return angle
        
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 180.0  # Default to straight
    
    def _get_person_center(self, landmarks) -> Tuple[float, float]:
        """
        Calculate the center position of a detected person
        
        Uses hip center as the reference point for tracking
        
        Returns:
            Tuple of (x, y) coordinates
        """
        try:
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            center_x = (left_hip.x + right_hip.x) / 2
            center_y = (left_hip.y + right_hip.y) / 2
            
            return (center_x, center_y)
        except Exception as e:
            print(f"Error getting person center: {e}")
            return (0.5, 0.5)  # Default to center of frame
    
    def _match_or_create_person(
        self, 
        person_center: Tuple[float, float], 
        next_person_id: int,
        max_persons: int
    ) -> Optional[int]:
        """
        Match detected person to existing tracked person or create new one
        
        Args:
            person_center: (x, y) position of detected person
            next_person_id: Next available person ID
            max_persons: Maximum number of persons to track
            
        Returns:
            Person ID or None if max_persons exceeded
        """
        # Try to match with existing persons
        best_match_id = None
        best_match_distance = float('inf')
        
        for person_id, last_position in self.active_persons.items():
            distance = self._calculate_distance(person_center, last_position)
            
            if distance < self.PERSON_DISTANCE_THRESHOLD and distance < best_match_distance:
                best_match_id = person_id
                best_match_distance = distance
        
        if best_match_id is not None:
            # Update position for matched person
            self.active_persons[best_match_id] = person_center
            return best_match_id
        else:
            # Create new person if under limit
            if len(self.active_persons) < max_persons:
                new_id = next_person_id
                self.active_persons[new_id] = person_center
                print(f"New person detected: person_{new_id} at position {person_center}")
                return new_id
            else:
                # Max persons reached, skip this detection
                return None
    
    def _calculate_distance(
        self, 
        pos1: Tuple[float, float], 
        pos2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two positions
        
        Returns:
            Distance as float
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
