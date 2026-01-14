#!/usr/bin/env python3
"""
Human Feedback Integration Module
Implements systems for integrating human feedback into coherence recalibration
"""

import sys
import os
import json
import time
import hashlib
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

@dataclass
class HumanFeedback:
    """Represents human feedback for coherence recalibration"""
    feedback_id: str
    user_id: str
    feedback_type: str  # coherence, entropy, flow, general
    message: str
    rating: int  # 0-10
    timestamp: float
    coherence_impact: float = 0.0  # Estimated impact on coherence
    processed: bool = False

class HumanFeedbackSystem:
    """
    Implements human feedback integration for coherence recalibration
    """
    
    def __init__(self, system_id: str = "quantum-currency-human-feedback"):
        self.system_id = system_id
        self.feedback_records: List[HumanFeedback] = []
        self.feedback_config = {
            "max_feedback_records": 1000,
            "coherence_weight": 0.3,
            "rating_weight": 0.7,
            "processing_interval": 300  # 5 minutes
        }
        self.last_processing_time = 0.0
    
    def submit_feedback(self, user_id: str, feedback_type: str, message: str, 
                       rating: int) -> Optional[str]:
        """
        Submit human feedback for coherence recalibration
        
        Args:
            user_id: ID of the user submitting feedback
            feedback_type: Type of feedback (coherence, entropy, flow, general)
            message: Feedback message
            rating: Rating (0-10)
            
        Returns:
            Feedback ID if successful, None otherwise
        """
        # Validate inputs
        if not user_id or not feedback_type or not message:
            print("Invalid feedback parameters")
            return None
        
        if rating < 0 or rating > 10:
            print("Invalid rating value")
            return None
        
        # Create feedback ID
        feedback_id = hashlib.sha256(f"{user_id}{feedback_type}{message}{time.time()}".encode()).hexdigest()[:32]
        
        # Create feedback record
        feedback = HumanFeedback(
            feedback_id=feedback_id,
            user_id=user_id,
            feedback_type=feedback_type,
            message=message,
            rating=rating,
            timestamp=time.time()
        )
        
        # Estimate coherence impact based on feedback type and rating
        feedback.coherence_impact = self._estimate_coherence_impact(feedback_type, rating)
        
        # Store feedback
        self.feedback_records.append(feedback)
        
        # Keep only recent feedback records
        if len(self.feedback_records) > self.feedback_config["max_feedback_records"]:
            self.feedback_records = self.feedback_records[-self.feedback_config["max_feedback_records"]:]
        
        return feedback_id
    
    def _estimate_coherence_impact(self, feedback_type: str, rating: int) -> float:
        """
        Estimate coherence impact based on feedback type and rating
        
        Args:
            feedback_type: Type of feedback
            rating: Rating (0-10)
            
        Returns:
            Estimated coherence impact (-1.0 to 1.0)
        """
        # Convert rating to impact factor (-1.0 to 1.0)
        impact_factor = (rating - 5) / 5.0  # -1.0 to 1.0
        
        # Apply type-specific weighting
        type_weights = {
            "coherence": 1.0,
            "entropy": 0.8,
            "flow": 0.6,
            "general": 0.5
        }
        
        weight = type_weights.get(feedback_type, 0.5)
        return impact_factor * weight
    
    def process_feedback(self) -> Dict[str, Any]:
        """
        Process human feedback to generate coherence recalibration recommendations
        
        Returns:
            Dictionary with processing results
        """
        current_time = time.time()
        
        # Check if it's time to process feedback
        if current_time - self.last_processing_time < self.feedback_config["processing_interval"]:
            return {"status": "skipped", "message": "Processing interval not reached"}
        
        self.last_processing_time = current_time
        
        # Get unprocessed feedback
        unprocessed_feedback = [f for f in self.feedback_records if not f.processed]
        
        if not unprocessed_feedback:
            return {"status": "no_data", "message": "No unprocessed feedback"}
        
        # Calculate weighted coherence impact
        total_weighted_impact = 0.0
        total_weight = 0.0
        
        for feedback in unprocessed_feedback:
            # Calculate weight based on recency and rating
            time_decay = max(0.1, 1.0 - (current_time - feedback.timestamp) / 86400)  # 24-hour decay
            rating_weight = feedback.rating / 10.0
            
            weight = time_decay * rating_weight
            weighted_impact = feedback.coherence_impact * weight
            
            total_weighted_impact += weighted_impact
            total_weight += weight
            
            # Mark as processed
            feedback.processed = True
        
        # Calculate average weighted impact
        average_impact = total_weighted_impact / total_weight if total_weight > 0 else 0.0
        
        # Generate recommendations based on impact
        recommendations = self._generate_recommendations(average_impact, len(unprocessed_feedback))
        
        return {
            "status": "success",
            "processed_feedback": len(unprocessed_feedback),
            "average_coherence_impact": average_impact,
            "recommendations": recommendations,
            "timestamp": current_time
        }
    
    def _generate_recommendations(self, average_impact: float, feedback_count: int) -> List[Dict[str, Any]]:
        """
        Generate coherence recalibration recommendations based on feedback impact
        
        Args:
            average_impact: Average coherence impact from feedback
            feedback_count: Number of feedback records processed
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # If significant positive impact, suggest amplifying coherence
        if average_impact > 0.3:
            recommendations.append({
                "type": "amplify_coherence",
                "priority": "high",
                "description": "Human feedback indicates strong positive coherence experience",
                "suggested_adjustment": average_impact * 0.1,
                "confidence": min(1.0, feedback_count / 50.0)  # Confidence based on feedback volume
            })
        
        # If significant negative impact, suggest coherence adjustments
        elif average_impact < -0.3:
            recommendations.append({
                "type": "adjust_coherence",
                "priority": "high",
                "description": "Human feedback indicates coherence issues",
                "suggested_adjustment": average_impact * 0.1,
                "confidence": min(1.0, feedback_count / 50.0)
            })
        
        # For moderate impact, suggest monitoring
        else:
            recommendations.append({
                "type": "monitor_coherence",
                "priority": "medium",
                "description": "Human feedback indicates stable coherence experience",
                "suggested_adjustment": 0.0,
                "confidence": min(1.0, feedback_count / 20.0)
            })
        
        return recommendations
    
    def get_feedback_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get feedback history
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of feedback records
        """
        # Return most recent feedback records
        recent_feedback = self.feedback_records[-limit:] if self.feedback_records else []
        
        return [{
            "feedback_id": f.feedback_id,
            "user_id": f.user_id,
            "feedback_type": f.feedback_type,
            "message": f.message,
            "rating": f.rating,
            "timestamp": f.timestamp,
            "coherence_impact": f.coherence_impact,
            "processed": f.processed
        } for f in recent_feedback]
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Get feedback summary statistics
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.feedback_records:
            return {"status": "no_data", "message": "No feedback records"}
        
        # Calculate statistics
        total_feedback = len(self.feedback_records)
        processed_feedback = len([f for f in self.feedback_records if f.processed])
        average_rating = np.mean([f.rating for f in self.feedback_records])
        
        # Count feedback by type
        type_counts = {}
        for f in self.feedback_records:
            if f.feedback_type not in type_counts:
                type_counts[f.feedback_type] = 0
            type_counts[f.feedback_type] += 1
        
        # Calculate average coherence impact
        avg_coherence_impact = np.mean([f.coherence_impact for f in self.feedback_records])
        
        return {
            "status": "success",
            "total_feedback": total_feedback,
            "processed_feedback": processed_feedback,
            "unprocessed_feedback": total_feedback - processed_feedback,
            "average_rating": float(average_rating),
            "feedback_by_type": type_counts,
            "average_coherence_impact": float(avg_coherence_impact),
            "timestamp": time.time()
        }

def demo_human_feedback_system():
    """Demonstrate human feedback system capabilities"""
    print("👤 Human Feedback Integration Demo")
    print("=" * 35)
    
    # Create feedback system
    feedback_system = HumanFeedbackSystem("demo-feedback-system")
    
    # Submit some feedback
    print("\n📝 Submitting Human Feedback:")
    
    feedback1_id = feedback_system.submit_feedback(
        user_id="user-001",
        feedback_type="coherence",
        message="The system feels very harmonious and balanced",
        rating=9
    )
    
    feedback2_id = feedback_system.submit_feedback(
        user_id="user-002",
        feedback_type="entropy",
        message="Noticed some instability in the flow",
        rating=4
    )
    
    feedback3_id = feedback_system.submit_feedback(
        user_id="user-003",
        feedback_type="general",
        message="Overall good experience with the system",
        rating=7
    )
    
    if feedback1_id and feedback2_id and feedback3_id:
        print("   Feedback submitted successfully")
    else:
        print("   Failed to submit feedback")
        return
    
    # Get feedback summary
    print("\n📊 Feedback Summary:")
    summary = feedback_system.get_feedback_summary()
    if summary["status"] == "success":
        print(f"   Total Feedback: {summary['total_feedback']}")
        print(f"   Average Rating: {summary['average_rating']:.1f}")
        print(f"   Average Coherence Impact: {summary['average_coherence_impact']:.3f}")
        print("   Feedback by Type:")
        for ftype, count in summary['feedback_by_type'].items():
            print(f"      {ftype}: {count}")
    
    # Get feedback history
    print("\n📋 Recent Feedback:")
    history = feedback_system.get_feedback_history(3)
    for feedback in history:
        print(f"   {feedback['user_id']}: {feedback['message'][:50]}... (Rating: {feedback['rating']})")
    
    # Process feedback
    print("\n⚙️  Processing Feedback:")
    processing_result = feedback_system.process_feedback()
    if processing_result["status"] == "success":
        print(f"   Processed {processing_result['processed_feedback']} feedback records")
        print(f"   Average Impact: {processing_result['average_coherence_impact']:.3f}")
        print("   Recommendations:")
        for rec in processing_result["recommendations"]:
            print(f"      {rec['type']}: {rec['description']}")
    else:
        print(f"   {processing_result['message']}")
    
    print("\n✅ Human feedback system demo completed!")

if __name__ == "__main__":
    demo_human_feedback_system()