#!/usr/bin/env python3
"""
Example script demonstrating how to use fall_detector_main.py
"""

import json
import subprocess
from pathlib import Path


def example_1_basic_detection():
    """Example 1: Basic fall detection on a video."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Fall Detection")
    print("=" * 80)
    
    video_path = "data/test_video.mp4"
    
    # Run detector
    result = subprocess.run(
        ['python', 'fall_detector_main.py',
         '--video', video_path,
         '--detector', 'ensemble'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    # Check exit code
    if result.returncode == 1:
        print("✓ Fall detected!")
    elif result.returncode == 0:
        print("✓ No fall detected")
    else:
        print(f"✗ Error: {result.stderr}")


def example_2_with_output_file():
    """Example 2: Save results to JSON file."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Save Results to File")
    print("=" * 80)
    
    video_path = "data/test_video.mp4"
    output_path = "results/example_output.json"
    
    # Run detector with output
    subprocess.run(
        ['python', 'fall_detector_main.py',
         '--video', video_path,
         '--detector', 'ensemble',
         '--output', output_path],
        capture_output=True
    )
    
    # Read and display results
    if Path(output_path).exists():
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        print(f"Video: {data['video']}")
        print(f"Detector: {data['detector']}")
        print(f"Prediction: {data['label']}")
        print(f"Confidence: {data['probability']:.2%}")
        
        if 'individual_predictions' in data:
            print("\nIndividual predictions:")
            for name, pred in data['individual_predictions'].items():
                status = 'FALL' if pred['prediction'] == 1 else 'NO FALL'
                print(f"  {name:12s}: {status:8s} ({pred['confidence']:.2%})")


def example_3_compare_detectors():
    """Example 3: Compare different detectors on same video."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Compare Detectors")
    print("=" * 80)
    
    video_path = "data/test_video.mp4"
    detectors = ['physics', 'temporal', 'geometry', 'ensemble']
    
    results = {}
    
    for detector in detectors:
        output_path = f"results/compare_{detector}.json"
        
        subprocess.run(
            ['python', 'fall_detector_main.py',
             '--video', video_path,
             '--detector', detector,
             '--output', output_path,
             '--max-frames', '64'],  # Faster
            capture_output=True
        )
        
        if Path(output_path).exists():
            with open(output_path, 'r') as f:
                data = json.load(f)
            results[detector] = data
    
    # Display comparison
    print(f"\n{'Detector':<12} {'Prediction':<10} {'Confidence':<12}")
    print("-" * 40)
    for detector, data in results.items():
        print(f"{detector:<12} {data['label']:<10} {data['probability']:>10.2%}")


def example_4_batch_processing():
    """Example 4: Process multiple videos."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Processing")
    print("=" * 80)
    
    # Example video list
    videos = [
        "data/video1.mp4",
        "data/video2.mp4",
        "data/video3.mp4",
    ]
    
    results = []
    
    for video_path in videos:
        if not Path(video_path).exists():
            print(f"Skipping {video_path} (not found)")
            continue
        
        output_path = f"results/{Path(video_path).stem}_result.json"
        
        print(f"Processing: {video_path}")
        subprocess.run(
            ['python', 'fall_detector_main.py',
             '--video', video_path,
             '--detector', 'ensemble',
             '--output', output_path],
            capture_output=True
        )
        
        if Path(output_path).exists():
            with open(output_path, 'r') as f:
                data = json.load(f)
            results.append(data)
    
    # Summary
    if results:
        falls = sum(1 for r in results if r['prediction'] == 1)
        print(f"\nProcessed {len(results)} videos")
        print(f"Falls detected: {falls}")
        print(f"No falls: {len(results) - falls}")


def example_5_python_api():
    """Example 5: Use as Python API (not subprocess)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Python API Usage")
    print("=" * 80)
    
    from fall_detector_main import FallDetectorMain
    
    # Initialize detector
    detector = FallDetectorMain(
        detector_type='ensemble',
        device='cpu'
    )
    
    # Process video
    video_path = Path("data/test_video.mp4")
    
    if video_path.exists():
        result = detector.process_video(video_path, max_frames=64)
        
        print(f"\nPrediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.2%}")
        
        if 'individual_predictions' in result:
            print("\nIndividual predictions:")
            for name, (pred, conf) in result['individual_predictions'].items():
                print(f"  {name}: {pred} ({conf:.2%})")
    else:
        print(f"Video not found: {video_path}")


def main():
    """Run all examples."""
    print("FALL DETECTOR - USAGE EXAMPLES")
    print()
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    try:
        # Run examples
        # example_1_basic_detection()  # Uncomment if you have a test video
        # example_2_with_output_file()
        # example_3_compare_detectors()
        # example_4_batch_processing()
        example_5_python_api()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have:")
        print("  1. Downloaded OpenPose models: python scripts/download_pose_models.py")
        print("  2. A test video at data/test_video.mp4")
        print("  3. Installed requirements: pip install -r requirements.txt")


if __name__ == '__main__':
    main()
