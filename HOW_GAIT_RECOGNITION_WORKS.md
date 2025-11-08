# Gait Recognition System - How It Works

## System Overview

This is **NOT** a simple person detection system. This is a **GAIT RECOGNITION** system that identifies users based on their unique walking patterns.

## How It Differs from Simple Detection

### ❌ Simple Person Detection (What we DON'T do)
- Just detects "a person" in the video
- Cannot identify WHO the person is
- Only gives you bounding boxes

### ✅ Gait Recognition (What we DO)
- Detects persons AND identifies which specific user they are
- Compares walking patterns against enrolled users in database
- Returns matched user ID, username, and confidence score

## The Complete Workflow

### Phase 1: User Enrollment

1. **User uploads walking video** → `POST /gait/user/upload-video`
2. **System detects the person** using YOLO (person detection)
3. **System extracts gait features** (512-dimensional embedding vector)
   - Analyzes walking pattern across multiple frames
   - Captures unique gait characteristics
   - Creates a "gait fingerprint" for the user
4. **Stores gait profile** in database linked to user account

### Phase 2: CCTV Recognition

1. **Admin uploads CCTV video** → `POST /gait/admin/upload-cctv-video`
2. **System detects all persons** in the video using YOLO
3. **For EACH detected person:**
   - Extracts their gait features (512-dim embedding)
   - **Compares against ALL enrolled users** in database
   - Uses cosine similarity to find best match
   - If similarity > threshold (0.6): User recognized ✅
   - If similarity < threshold: Unknown person ❌
4. **Creates annotated video** with:
   - Green boxes: Recognized users (with name + confidence)
   - Red boxes: Unknown persons
5. **Stores results** with detailed detection logs

## Database Schema

```
GaitProfile:
- user_id → Links to specific user
- embedding → 512-dim gait feature vector
- video_path → Original enrollment video

GaitRecognitionLog:
- admin_id → Who uploaded the CCTV
- total_persons_detected → Count of all persons
- total_recognized → Count of matched users
- processing_status → pending/processing/completed/failed

GaitDetection:
- log_id → Links to recognition log
- person_index → Which person (0, 1, 2...)
- matched_user_id → Which user was recognized (NULL if unknown)
- confidence_score → Similarity score (0-1)
- is_recognized → TRUE/FALSE
```

## API Response Example

When admin retrieves recognition results:

```json
{
  "id": 1,
  "total_persons_detected": 3,
  "total_recognized": 2,
  "processing_status": "completed",
  "detections": [
    {
      "person_index": 0,
      "matched_user_id": 5,
      "matched_username": "john_doe",
      "matched_full_name": "John Doe",
      "confidence_score": 0.87,
      "is_recognized": true
    },
    {
      "person_index": 1,
      "matched_user_id": 8,
      "matched_username": "jane_smith",
      "matched_full_name": "Jane Smith",
      "confidence_score": 0.92,
      "is_recognized": true
    },
    {
      "person_index": 2,
      "matched_user_id": null,
      "matched_username": null,
      "matched_full_name": null,
      "confidence_score": 0.42,
      "is_recognized": false
    }
  ]
}
```

## Key Points

1. **Person Detection** (YOLO) is only the FIRST step
2. **Gait Recognition** is the MAIN feature
3. System compares each detected person against ALL enrolled users
4. Returns WHICH specific user was detected, not just "a person"
5. Confidence score shows how certain the match is
6. Threshold (0.6) determines if match is accepted or rejected

## Processing Flow Diagram

```
CCTV Video Upload
      ↓
Person Detection (YOLO)
      ↓
For each person:
      ↓
Extract Gait Features (512-dim vector)
      ↓
Compare with ALL users in database
      ↓
Find best match using cosine similarity
      ↓
If similarity >= 0.6 → RECOGNIZED (return user details)
If similarity < 0.6  → UNKNOWN (no match)
      ↓
Save results to database
      ↓
Generate annotated video with names
```

## Important Notes

- **Number of persons detected**: Physical persons in video (from YOLO)
- **Number recognized**: How many matched enrolled users (from gait recognition)
- **Unknown persons**: Detected but not in database OR below confidence threshold
- Every detection is checked against the ENTIRE user database
- The system learns each user's unique gait during enrollment
- Recognition happens by comparing gait patterns, not faces or appearance

## Current Limitations

- Using fallback feature extractor (appearance-based) instead of true OpenGait model
- For production accuracy, need to integrate actual GaitBase model
- Current features work for demo but may have lower accuracy than true gait analysis

## To Improve Accuracy

1. Download GaitBase model from OpenGait repository
2. Place in `models/gaitbase_model.pt`
3. System will automatically use it for feature extraction
4. Will provide much better gait-based recognition vs current appearance-based fallback
