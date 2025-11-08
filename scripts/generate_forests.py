"""
Generate Wellness Forest data for all users
Calculates forests based on existing wellness metrics
"""
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.models.user import User
from app.services.forest_calculator import ForestCalculator


def generate_forests():
    """Generate forests for all users"""
    db = SessionLocal()
    forest_calculator = ForestCalculator()
    
    try:
        # Get all users
        users = db.query(User).all()
        
        if not users:
            print("❌ No users found in database")
            return
        
        print(f"🌲 Generating wellness forests for {len(users)} users...\n")
        
        for user in users:
            print(f"Processing user: {user.username} (ID: {user.id})")
            
            # Calculate and create/update forest
            forest = forest_calculator.calculate_and_update_forest(db, user.id)
            
            print(f"  ✅ Forest created:")
            print(f"     Total trees: {forest.total_trees}")
            print(f"     Healthy: {forest.healthy_trees}, Growing: {forest.growing_trees}, Wilting: {forest.wilting_trees}, Dead: {forest.dead_trees}")
            print(f"     Forest health: {forest.forest_health_score}%")
            print(f"     Season: {forest.season}, Weather: {forest.weather}, Time: {forest.time_of_day}")
            print(f"     Special features: Flowers={forest.has_flowers}, Birds={forest.has_birds}, Butterflies={forest.has_butterflies}")
            print(f"     Stream={forest.has_stream}, Bench={forest.has_bench}")
            print()
        
        print("✅ All wellness forests generated successfully!")
        
    except Exception as e:
        print(f"❌ Error generating forests: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    generate_forests()
