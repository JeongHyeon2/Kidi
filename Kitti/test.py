import os
import sys
import numpy as np
import cv2
from speed_estimation import VehicleSpeedEstimator, process_kitti_sequence

def test_with_yolo(video_path, calib_path, oxts_path, output_path):
    """
    Enhanced version using YOLO for better vehicle detection
    """
    try:
        # Try to import YOLO
        from ultralytics import YOLO
        use_yolo = True
        model = YOLO('yolov8n.pt')  # You can use yolov7.pt if available
    except ImportError:
        print("YOLO not available, using simple detection method")
        use_yolo = False
    
    class EnhancedVehicleSpeedEstimator(VehicleSpeedEstimator):
        def __init__(self, calib_path, use_yolo=False, yolo_model=None):
            super().__init__(calib_path)
            self.use_yolo = use_yolo
            self.yolo_model = yolo_model
            # Ensure calibration is loaded
            if not hasattr(self, 'K') or self.K is None:
                print("Warning: Calibration not properly loaded")
            
        def detect_vehicles(self, frame):
            if self.use_yolo and self.yolo_model:
                # Use YOLO for detection
                results = self.yolo_model(frame)
                detections = []
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Filter for vehicle classes (car, truck, bus, motorcycle)
                            if box.cls in [2, 5, 7, 3]:  # COCO class IDs
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                detections.append([int(x1), int(y1), int(x2), int(y2)])
                
                return detections
            else:
                # Fall back to parent method
                return super().detect_vehicles(frame)
    
    # Initialize enhanced estimator
    estimator = EnhancedVehicleSpeedEstimator(
        calib_path, 
        use_yolo=use_yolo, 
        yolo_model=model if use_yolo else None
    )
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load ground truth
    gt_velocities = estimator.load_ground_truth_velocities(oxts_path, frame_count)
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height + 400))
    
    frame_id = 0
    all_errors = []
    
    print(f"Processing {frame_count} frames...")
    print(f"FPS: {fps}")
    print(f"Using YOLO: {use_yolo}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track
        detections = estimator.detect_vehicles(frame)
        tracked = estimator.track_vehicles(detections, frame_id)
        
        # Get ground truth
        gt_velocity = gt_velocities[frame_id] if frame_id < len(gt_velocities) else 0
        
        # Visualize
        result_frame = estimator.visualize_results(frame, tracked, gt_velocity, frame_id)
        out.write(result_frame)
        
        # Calculate error
        if tracked:
            est_velocities = [estimator.estimate_velocity(track_id, fps) for _, track_id in tracked]
            if est_velocities:
                avg_est = np.mean(est_velocities)
                error = abs(gt_velocity - avg_est)
                all_errors.append(error)
        
        frame_id += 1
        
        if frame_id % 50 == 0:
            current_mae = np.mean(all_errors) if all_errors else 0
            print(f"Frame {frame_id}/{frame_count} - Current MAE: {current_mae:.2f} km/h")
    
    # Final results
    mae = np.mean(all_errors) if all_errors else 0
    rmse = np.sqrt(np.mean([e**2 for e in all_errors])) if all_errors else 0
    
    print(f"\n=== Final Results ===")
    print(f"Total frames processed: {frame_id}")
    print(f"Mean Absolute Error: {mae:.2f} km/h")
    print(f"Root Mean Square Error: {rmse:.2f} km/h")
    print(f"Output saved to: {output_path}")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

import os
import sys

if __name__ == "__main__":
    # KITTI dataset paths
    video_path = "output_0026_image02.mp4"
    calib_cam_path = "2011_09_29_calib/calib_cam_to_cam.txt"
    oxts_path = "2011_09_29_drive_0026_sync/oxts/"
    output_path = "ego_speed_estimation_output.mp4"
    
    print("=== KITTI Ego Vehicle Speed Estimation ===")
    print(f"Video: {video_path}")
    print(f"Calibration: {calib_cam_path}")
    print(f"Ground truth: {oxts_path}")
    
    # Check files
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        sys.exit(1)
    
    if not os.path.exists(calib_cam_path):
        print(f"Error: Calibration file {calib_cam_path} not found!")
        sys.exit(1)
    
    if not os.path.exists(oxts_path):
        print(f"Error: OXTS directory {oxts_path} not found!")
        sys.exit(1)
    
    # Import and run ego speed estimation
    from speed_estimation import process_kitti_ego_speed
    
    try:
        process_kitti_ego_speed(video_path, calib_cam_path, oxts_path, output_path)
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()