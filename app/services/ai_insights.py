"""
AI Insights Service using Google Gemini
Generates personalized wellness insights for users
"""
from google import genai
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from app.core.config import settings
from app.models.posture import WellnessMetrics, AttritionRisk


class AIInsightsGenerator:
    """Generate AI-powered wellness insights using Gemini"""
    
    def __init__(self):
        """Initialize Gemini API"""
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    def generate_wellness_insights(
        self,
        user_name: str,
        recent_metrics: List[WellnessMetrics],
        latest_risk: Optional[AttritionRisk] = None
    ) -> List[str]:
        """
        Generate 8 personalized wellness insights based on user data
        
        Args:
            user_name: User's name
            recent_metrics: List of recent wellness metrics (last 30 days)
            latest_risk: Latest attrition risk assessment
        
        Returns:
            List of 8 insight points
        """
        if not recent_metrics:
            return self._get_default_insights()
        
        # Prepare data summary for Gemini
        data_summary = self._prepare_data_summary(user_name, recent_metrics, latest_risk)
        
        # Create prompt for Gemini
        prompt = self._create_prompt(data_summary)
        
        try:
            # Generate insights using Gemini with the latest API
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            insights = self._parse_insights(response.text)
            
            # Ensure we have exactly 8 insights
            if len(insights) < 8:
                insights.extend(self._get_default_insights()[len(insights):8])
            
            return insights[:8]
        
        except Exception as e:
            print(f"Error generating AI insights: {e}")
            return self._get_default_insights()
    
    def _prepare_data_summary(
        self,
        user_name: str,
        recent_metrics: List[WellnessMetrics],
        latest_risk: Optional[AttritionRisk]
    ) -> Dict:
        """Prepare wellness data summary for AI analysis"""
        
        # Calculate averages
        avg_ergonomic = sum(m.ergonomic_score for m in recent_metrics) / len(recent_metrics)
        avg_posture = sum(m.posture_quality_score for m in recent_metrics) / len(recent_metrics)
        avg_activity = sum(m.activity_level_score for m in recent_metrics) / len(recent_metrics)
        
        # Get latest metrics
        latest = recent_metrics[-1] if recent_metrics else None
        
        # Calculate trends
        if len(recent_metrics) >= 7:
            first_week = recent_metrics[:7]
            last_week = recent_metrics[-7:]
            
            first_avg = sum(m.ergonomic_score for m in first_week) / len(first_week)
            last_avg = sum(m.ergonomic_score for m in last_week) / len(last_week)
            
            ergonomic_trend = "improving" if last_avg > first_avg + 5 else "declining" if last_avg < first_avg - 5 else "stable"
        else:
            ergonomic_trend = "stable"
        
        # Count posture issues
        forward_head_days = sum(1 for m in recent_metrics if m.forward_head_percent and m.forward_head_percent > 30)
        slouched_days = sum(1 for m in recent_metrics if m.slouched_percent and m.slouched_percent > 30)
        
        # Activity patterns
        sitting_avg = sum(m.sitting_time_minutes or 0 for m in recent_metrics) / len(recent_metrics)
        break_avg = sum(m.break_count or 0 for m in recent_metrics) / len(recent_metrics)
        
        # Stress and engagement
        stress_levels = [m.stress_level for m in recent_metrics if m.stress_level]
        high_stress_days = sum(1 for s in stress_levels if s in ['high', 'very_high'])
        
        engagement_levels = [m.engagement_level for m in recent_metrics if m.engagement_level]
        low_engagement_days = sum(1 for e in engagement_levels if e in ['disengaged', 'neutral'])
        
        return {
            'user_name': user_name,
            'period_days': len(recent_metrics),
            'avg_ergonomic_score': round(avg_ergonomic, 1),
            'avg_posture_score': round(avg_posture, 1),
            'avg_activity_score': round(avg_activity, 1),
            'ergonomic_trend': ergonomic_trend,
            'forward_head_days': forward_head_days,
            'slouched_days': slouched_days,
            'avg_sitting_minutes': round(sitting_avg, 0),
            'avg_breaks_per_day': round(break_avg, 1),
            'high_stress_days': high_stress_days,
            'low_engagement_days': low_engagement_days,
            'current_stress': latest.stress_level if latest else None,
            'current_mood': latest.mood_estimate if latest else None,
            'current_engagement': latest.engagement_level if latest else None,
            'neck_pain_risk': latest.neck_pain_risk if latest else None,
            'back_pain_risk': latest.back_pain_risk if latest else None,
            'risk_category': latest_risk.risk_category if latest_risk else None,
            'risk_score': latest_risk.overall_risk_score if latest_risk else None,
        }
    
    def _create_prompt(self, data_summary: Dict) -> str:
        """Create prompt for Gemini AI"""
        return f"""You are a wellness and ergonomics expert analyzing workplace health data. Based on the following wellness data for {data_summary['user_name']}, generate exactly 8 concise, actionable, and personalized wellness insights.

Data Summary (last {data_summary['period_days']} days):
- Average Ergonomic Score: {data_summary['avg_ergonomic_score']}/100
- Average Posture Quality: {data_summary['avg_posture_score']}/100
- Average Activity Level: {data_summary['avg_activity_score']}/100
- Ergonomic Trend: {data_summary['ergonomic_trend']}
- Days with Forward Head Posture (>30%): {data_summary['forward_head_days']}
- Days with Slouching (>30%): {data_summary['slouched_days']}
- Average Sitting Time: {data_summary['avg_sitting_minutes']} minutes/day
- Average Breaks: {data_summary['avg_breaks_per_day']} per day
- High Stress Days: {data_summary['high_stress_days']}
- Low Engagement Days: {data_summary['low_engagement_days']}
- Current Stress Level: {data_summary['current_stress']}
- Current Mood: {data_summary['current_mood']}
- Current Engagement: {data_summary['current_engagement']}
- Neck Pain Risk: {data_summary['neck_pain_risk']}
- Back Pain Risk: {data_summary['back_pain_risk']}
- Attrition Risk Category: {data_summary['risk_category']}
- Attrition Risk Score: {data_summary['risk_score']}

Generate exactly 8 insights that:
1. Are specific and actionable (not generic advice)
2. Reference the actual data points when relevant
3. Provide concrete recommendations
4. Cover different aspects: posture, activity, stress, engagement, pain prevention
5. Are positive and encouraging while being realistic
6. Each insight should be 1-2 sentences, maximum 150 characters
7. Start each insight with a number (1., 2., etc.)
8. Focus on improvements and preventive measures

Format: Return only the numbered list, one insight per line."""

    def _parse_insights(self, response_text: str) -> List[str]:
        """Parse Gemini response into list of insights"""
        lines = response_text.strip().split('\n')
        insights = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (1., 2., *, -, etc.)
            for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '*', '-', '•']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            
            if line:
                insights.append(line)
        
        return insights
    
    def _get_default_insights(self) -> List[str]:
        """Return default insights when AI generation fails"""
        return [
            "Monitor your posture regularly throughout the day to prevent strain and discomfort.",
            "Take regular breaks every hour to reduce muscle tension and improve circulation.",
            "Adjust your workstation ergonomics to maintain proper alignment and reduce injury risk.",
            "Stay hydrated and maintain good nutrition to support overall wellness and energy levels.",
            "Practice stress management techniques like deep breathing during work hours.",
            "Engage in regular physical activity to counterbalance sedentary work time.",
            "Ensure adequate sleep and recovery to optimize workplace performance and mood.",
            "Build social connections with colleagues to enhance engagement and job satisfaction."
        ]


# Create singleton instance
ai_insights_generator = AIInsightsGenerator()
