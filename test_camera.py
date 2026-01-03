import cv2
import time

print("Testing camera access...")

# Try different indices and backends
for camera_idx in [0, 1, 2]:
    print(f"\n--- Testing camera index {camera_idx} ---")
    
    # Try with AVFoundation backend
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_AVFOUNDATION)
    
    if cap.isOpened():
        print(f"✓ Camera {camera_idx} opened with AVFoundation")
        
        # Try to grab a few frames
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"  ✓ Frame {i+1}: Success - shape {frame.shape}")
                break
            else:
                print(f"  ✗ Frame {i+1}: Failed")
            time.sleep(0.1)
        
        cap.release()
    else:
        print(f"✗ Camera {camera_idx} failed to open")

print("\nDone!")
