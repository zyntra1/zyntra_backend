# Posture Detection & Wellness Monitoring System

## 📋 Overview

This system provides comprehensive posture detection, wellness monitoring, and attrition risk prediction for employees using AI/ML technologies. It analyzes video feeds (CCTV or user uploads) to extract posture data, generate wellness metrics, and predict employee attrition risk.

## 🎯 Features

### 1. **Posture Detection (MediaPipe)**
- Real-time pose estimation using MediaPipe
- Classifies posture types:
  - ✅ Good posture
  - ⚠️ Forward head
  - ⚠️ Slouched
  - ⚠️ Leaning
  - ⚠️ Moderate
- Calculates skeletal metrics:
  - Neck angle
  - Shoulder symmetry
  - Spine alignment score
- Activity detection (sitting, standing, walking)

### 2. **Wellness Monitoring**
- **Ergonomic Score** (0-100): Overall ergonomic health
- **Posture Quality Score** (0-100): Posture alignment quality
- **Activity Level Score** (0-100): Movement and activity patterns
- **Stress Level**: low, moderate, high, very_high
- **Fatigue Indicator** (0-1): Work-related fatigue
- **Engagement Level**: disengaged, neutral, engaged, highly_engaged
- **Mood Estimate**: negative, neutral, positive

### 3. **Pain Risk Assessment**
- Neck pain risk (0-1)
- Back pain risk (0-1)
- Shoulder pain risk (0-1)

### 4. **Behavioral Analytics**
- Break compliance scoring
- Fidgeting frequency
- Position change tracking
- Sitting/standing/walking time breakdown

### 5. **Attrition Risk Prediction**
- ML-based risk scoring (0-100)
- Risk categories: low, moderate, high, critical
- Estimated weeks to potential attrition
- Top contributing risk factors
- Personalized intervention recommendations

### 6. **Baseline Profiling**
- Learns individual user patterns over 2-4 weeks
- Establishes normal posture distributions
- Enables anomaly detection
- Tracks deviations from baseline

---

## 🗂️ Database Models

### PostureSnapshot
Stores individual posture measurements from video frames.

**Fields:**
- `user_id`: Foreign key to User
- `timestamp`: When the snapshot was captured
- `posture_type`: Classification (good, forward_head, slouched, etc.)
- `confidence`: Model confidence (0-1)
- `neck_angle`, `shoulder_symmetry`, `spine_score`: Skeletal metrics
- `activity`: sitting, standing, walking
- `keypoints_json`: Full MediaPipe landmarks (33 points)
- `session_id`: Groups snapshots from same video
- `video_source`: CCTV camera ID or upload source

### WellnessMetrics
Aggregated daily/hourly wellness data.

**Fields:**
- `user_id`: Foreign key to User
- `date`, `period_type`: Time period (hourly, daily, weekly)
- `ergonomic_score`, `posture_quality_score`, `activity_level_score`
- `stress_level`, `fatigue_indicator`, `engagement_level`, `mood_estimate`
- `good_posture_percent`, `slouched_percent`, etc.
- `sitting_time_minutes`, `standing_time_minutes`, `walking_time_minutes`
- `break_count`, `break_compliance_score`
- `neck_pain_risk`, `back_pain_risk`, `shoulder_pain_risk`

### AttritionRisk
Employee attrition risk predictions.

**Fields:**
- `user_id`: Foreign key to User
- `prediction_date`: When prediction was made
- `overall_risk_score` (0-100)
- `risk_category`: low, moderate, high, critical
- `estimated_weeks_to_attrition`
- `top_risk_factors`: JSON array of contributing factors
- `wellness_score_trend`, `stress_trend`, `engagement_trend`
- `recommended_interventions`: List of suggested actions

### PostureBaseline
User-specific baseline profiles.

**Fields:**
- `user_id`: Foreign key to User (unique)
- `baseline_start_date`, `baseline_end_date`
- `is_complete`: Whether baseline is fully established
- `avg_neck_angle`, `avg_shoulder_symmetry`, `avg_spine_score`
- `typical_posture_distribution`: JSON with normal distributions
- `avg_break_frequency`, `avg_daily_sitting_time`
- Anomaly detection thresholds

---

## 🔌 API Endpoints

### Admin Endpoints

#### 1. **POST /posture/analyze-cctv-demo**
Analyzes CCTV video for posture detection (demo mode).

**Request:**
- Multipart form-data with video file
- Requires admin authentication

**Response:**
```json
{
  "video_filename": "office_cctv.mp4",
  "analysis_duration_seconds": 45.2,
  "total_frames_processed": 450,
  "fps": 30,
  "persons_detected": 3,
  "person_analyses": [
    {
      "person_id": "Person 1",
      "total_frames_detected": 150,
      "avg_ergonomic_score": 72.5,
      "avg_posture_quality": 68.3,
      "dominant_posture": "good",
      "posture_distribution": {
        "good": 65.2,
        "forward_head": 20.1,
        "slouched": 14.7
      },
      "stress_level": "moderate",
      "fatigue_indicator": 0.42,
      "mood_estimate": "neutral",
      "engagement_level": "engaged",
      "activity_distribution": {
        "sitting": 78.5,
        "standing": 15.3,
        "walking": 6.2
      },
      "neck_pain_risk": 0.25,
      "back_pain_risk": 0.18,
      "overall_wellness_score": 71.8,
      "position_changes": 12,
      "fidgeting_frequency": 2.5,
      "sample_snapshots": [...]
    }
  ],
  "overall_avg_wellness": 69.4,
  "high_risk_count": 1,
  "session_id": "abc-123-xyz",
  "processed_at": "2025-11-08T12:00:00Z"
}
```

#### 2. **GET /wellness/dashboard** (Admin)
Get aggregated wellness dashboard for all employees.

**Response:**
```json
{
  "total_employees": 5,
  "avg_wellness_score": 65.3,
  "avg_ergonomic_score": 67.2,
  "avg_engagement": 58.4,
  "low_risk_count": 3,
  "moderate_risk_count": 2,
  "high_risk_count": 0,
  "critical_risk_count": 0,
  "employees": [
    {
      "user_id": 2,
      "username": "adhithya",
      "full_name": "Adhithya P",
      "current_wellness_score": 72.5,
      "wellness_trend": "improving",
      "stress_level": "low",
      "attrition_risk": "low",
      "attrition_risk_score": 18.0,
      "last_updated": "2025-11-08T12:00:00Z"
    }
  ],
  "wellness_distribution": {
    "excellent": 1,
    "good": 2,
    "fair": 2,
    "poor": 0
  },
  "last_updated": "2025-11-08T12:00:00Z"
}
```

### User Endpoints

#### 3. **GET /wellness/dashboard** (User)
Get personal wellness dashboard.

**Response:**
```json
{
  "user_id": 2,
  "username": "adhithya",
  "today_ergonomic_score": 72.5,
  "today_stress_level": "low",
  "today_mood": "neutral",
  "wellness_trend_7d": [...],
  "avg_wellness_7d": 70.3,
  "avg_wellness_30d": 68.5,
  "wellness_change_percent": 2.6,
  "current_risk": {
    "overall_risk_score": 18.0,
    "risk_category": "low",
    "top_risk_factors": [...],
    "recommended_interventions": [...]
  },
  "recommendations": [
    "Great job maintaining good posture!",
    "Consider taking more breaks"
  ],
  "posture_quality_trend": "improving",
  "top_posture_issues": ["Forward head posture"]
}
```

---

## 🚀 Setup & Usage

### 1. Install Dependencies
```bash
source zyntra_venv/bin/activate
# Already installed from requirements.txt:
# mediapipe, opencv-python, xgboost, scikit-learn, faker, numpy
```

### 2. Initialize Database
```bash
python init_db.py
```

This creates all necessary tables including:
- `posture_snapshots`
- `wellness_metrics`
- `attrition_risks`
- `posture_baselines`

### 3. Generate Demo Data
```bash
python scripts/generate_fake_wellness_data.py
```

Generates 30 days of fake wellness data for users with IDs 2-6:
- 270 posture snapshots per user (9 per day)
- 30 daily wellness metrics per user
- Baseline profiles
- Attrition risk predictions

### 4. Start the Server
```bash
./start.sh
# or
uvicorn main:app --reload
```

### 5. Test the API
Access Swagger docs at: `http://localhost:8000/docs`

**Test CCTV Analysis:**
1. Login as admin
2. Navigate to `/posture/analyze-cctv-demo`
3. Upload a video file (MP4, AVI, etc.)
4. Review the comprehensive analysis

---

## 🧪 Testing Demo

### Upload Test Video
Use any video with people visible (webcam recording, CCTV footage, etc.).

```bash
# Example using curl
curl -X POST "http://localhost:8000/posture/analyze-cctv-demo" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -F "file=@test_video.mp4"
```

### Check User Wellness
```bash
# Get user dashboard
curl -X GET "http://localhost:8000/wellness/dashboard" \
  -H "Authorization: Bearer YOUR_USER_TOKEN"
```

### Check Admin Dashboard
```bash
# Get all employees wellness
curl -X GET "http://localhost:8000/wellness/dashboard" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

---

## 📊 How It Works

### Posture Analysis Pipeline

1. **Video Upload** → Admin uploads CCTV video
2. **Frame Sampling** → Sample frames at 1 FPS (configurable)
3. **Pose Detection** → MediaPipe detects 33 body landmarks
4. **Metric Calculation** → Compute neck angle, spine score, shoulder symmetry
5. **Posture Classification** → Rule-based classification (good, forward_head, etc.)
6. **Activity Detection** → Classify as sitting, standing, or walking
7. **Person Tracking** → Track multiple persons across frames
8. **Wellness Scoring** → Aggregate metrics into wellness scores
9. **Report Generation** → Return comprehensive analysis

### Wellness Scoring Algorithm

**Ergonomic Score** (0-100):
```
score = Σ (posture_type_percentage × posture_weight)
  where:
    good = 1.0
    moderate = 0.7
    forward_head = 0.4
    slouched = 0.3
    leaning = 0.5
```

**Stress Level** (low to very_high):
```
Based on:
  - Poor posture percentage
  - Shoulder symmetry variation (tension indicator)
```

**Fatigue Indicator** (0-1):
```
Compare first-half vs second-half posture quality
Degradation indicates fatigue accumulation
```

### Attrition Risk Prediction

**Risk Factors Considered:**
1. Low wellness score (< 50)
2. Declining wellness trend (slope < -2)
3. High stress levels (> 80)
4. Low engagement (< 40)
5. High fatigue (> 0.7)
6. Poor break compliance (< 40)

**Risk Score Calculation:**
```
risk_score = 
  wellness_penalty (max 30) +
  trend_penalty (max 20) +
  stress_penalty (max 20) +
  engagement_penalty (max 15) +
  fatigue_penalty (max 10) +
  break_penalty (max 5)
```

**Risk Categories:**
- **Low**: 0-29 → No immediate concern
- **Moderate**: 30-49 → Monitor closely
- **High**: 50-74 → Intervention recommended
- **Critical**: 75-100 → Immediate action required

---

## 🎯 Key Insights

### For Employees (Users)
- View personal wellness trends
- Understand posture quality over time
- Track stress and fatigue levels
- Receive personalized recommendations
- Monitor break compliance
- See pain risk assessments

### For Admins/HR
- Monitor team-wide wellness
- Identify at-risk employees
- Detect early signs of burnout
- Optimize ergonomic interventions
- Track wellness improvements
- Generate compliance reports

---

## 🔮 Future Enhancements

1. **Real-time CCTV Integration**
   - Continuous monitoring
   - Live alerts for poor posture
   - Automated reminders

2. **Advanced ML Models**
   - Deep learning for posture classification
   - Personalized attrition models
   - Anomaly detection refinement

3. **Wearable Integration**
   - Smartwatch data fusion
   - Heart rate variability tracking
   - Sleep quality correlation

4. **Gamification**
   - Posture challenges
   - Wellness leaderboards
   - Achievement badges

5. **Predictive Analytics**
   - Forecast wellness trends
   - Seasonal pattern detection
   - Department-level insights

---

## 📖 API Documentation

Full API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🛡️ Privacy & Security

- All posture data is anonymized in demo mode
- User-specific data requires authentication
- Role-based access control (admin vs user)
- Video files not stored permanently
- Keypoints stored as JSON (no images/videos in DB)

---

## 📞 Support

For questions or issues:
- Check `/docs` for API details
- Review database models in `app/models/posture.py`
- Inspect service logic in `app/services/`

---

## 🏆 Demo Credentials

**Admin Login:**
- Email: `admin@zyntra.com`
- Password: `changeme123`

**User Logins:**
- adhithya: `adhithya@zyntra.com` / `user123`
- afish: `afish@zyntra.com` / `user123`
- amal: `amal@zyntra.com` / `user123`
- arpit: `arpit@zyntra.com` / `user123`
- munazir: `munazir@zyntra.com` / `user123`

---

## ✅ System Health Check

Test the system:

```bash
# Check posture service health
curl http://localhost:8000/posture/health

# Expected response:
{
  "status": "healthy",
  "service": "posture_detection",
  "version": "1.0.0",
  "mediapipe_available": true
}
```

---

**Built with ❤️ using MediaPipe, XGBoost, FastAPI, and PostgreSQL**
