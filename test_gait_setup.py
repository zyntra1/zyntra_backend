#!/usr/bin/env python3
"""
Test script for Gait Recognition System
Verifies all components are properly installed and configured
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        from sklearn.metrics.pairwise import cosine_similarity
        print("✓ Scikit-learn")
        
        import aiofiles
        print("✓ AIOFiles")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_directories():
    """Test required directories exist"""
    print("\nTesting directories...")
    required_dirs = [
        'models',
        'storage',
        'storage/user_videos',
        'storage/cctv_videos',
        'storage/processed_videos',
        'app/services',
        'app/models',
        'app/routers',
        'app/schemas',
        'app/utils'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist


def test_model_files():
    """Check for model files"""
    print("\nChecking for model files...")
    
    # Check for GaitBase model
    gait_model_path = "models/gaitbase_model.pt"
    if os.path.exists(gait_model_path):
        print(f"✓ GaitBase model found at {gait_model_path}")
        print(f"  Size: {os.path.getsize(gait_model_path) / (1024*1024):.2f} MB")
    else:
        print(f"⚠ GaitBase model NOT found at {gait_model_path}")
        print("  System will use fallback feature extractor")
        print("  For production use, download GaitBase model from:")
        print("  https://github.com/ShiqiYu/OpenGait")
    
    return True


def test_yolo():
    """Test YOLO model download"""
    print("\nTesting YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8 model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading YOLO: {e}")
        return False


def test_database_models():
    """Test database models can be imported"""
    print("\nTesting database models...")
    try:
        from app.models.gait import GaitProfile, GaitRecognitionLog, GaitDetection
        print("✓ Gait models imported successfully")
        
        from app.models.user import User, Admin
        print("✓ User models imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error importing models: {e}")
        return False


def test_services():
    """Test service modules"""
    print("\nTesting service modules...")
    try:
        from app.services.gait_processor import GaitProcessor, serialize_embedding, deserialize_embedding
        print("✓ GaitProcessor imported successfully")
        
        # Test initialization
        processor = GaitProcessor()
        print(f"✓ GaitProcessor initialized (device: {processor.device})")
        
        return True
    except Exception as e:
        print(f"✗ Error loading services: {e}")
        return False


def test_routers():
    """Test router modules"""
    print("\nTesting routers...")
    try:
        from app.routers.gait import router
        print("✓ Gait router imported successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading routers: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Gait Recognition System - Component Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Directories", test_directories()))
    results.append(("Model Files", test_model_files()))
    results.append(("YOLO", test_yolo()))
    results.append(("Database Models", test_database_models()))
    results.append(("Services", test_services()))
    results.append(("Routers", test_routers()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nNext steps:")
        print("1. Obtain GaitBase model from OpenGait repository")
        print("2. Place model file in models/gaitbase_model.pt")
        print("3. Start the server: python main.py")
        print("4. Visit http://localhost:8000/docs for API documentation")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
