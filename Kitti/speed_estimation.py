import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import json

class EgoVehicleSpeedEstimator:
    def __init__(self, calib_path):
        """
        Initialize with KITTI calibration data for ego vehicle speed estimation
        """
        self.load_calibration(calib_path)
        self.prev_frame = None
        self.frame_velocities = []
        
    def load_calibration(self, calib_path):
        """Load KITTI calibration parameters"""
        self.K = None
        
        with open(calib_path, 'r') as f:
            for line in f:
                if line.startswith('P_rect_02:'):  # Left color camera projection matrix
                    P2 = np.array(line.split()[1:13], dtype=np.float32).reshape(3, 4)
                    self.K = P2[:3, :3]  # Camera intrinsic matrix
                    print(f"Loaded camera matrix K:\n{self.K}")
        
        if self.K is None:
            # From KITTI calibration file: P_rect_02
            self.K = np.array([[7.183351e+02, 0.000000e+00, 6.003891e+02],
                               [0.000000e+00, 7.183351e+02, 1.815122e+02],
                               [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=np.float32)
        
        # Camera parameters
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        
        # For KITTI, camera height is approximately 1.65m
        self.camera_height = 1.65
        
    def detect_features(self, frame):
        """
        Detect features for optical flow
        Using Shi-Tomasi corner detection
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Parameters for Shi-Tomasi corner detection
        feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        
        # Detect corners - focus on lower part of image (road region)
        mask = np.zeros_like(gray)
        h, w = gray.shape
        # Focus on road region (lower 2/3 of image)
        mask[h//3:, :] = 255
        
        corners = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
        
        return corners, gray
    
    def calculate_optical_flow(self, prev_gray, curr_gray, prev_pts):
        """
        Calculate optical flow using Lucas-Kanade method
        """
        if prev_pts is None or len(prev_pts) == 0:
            return None, None, None
        
        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Calculate optical flow
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **lk_params
        )
        
        # Select good points
        if next_pts is not None:
            good_old = prev_pts[status == 1]
            good_new = next_pts[status == 1]
            return good_old, good_new, status
        
        return None, None, None
    
    def estimate_ego_motion(self, old_pts, new_pts):
        """
        Estimate ego vehicle motion from optical flow
        Using essential matrix decomposition
        """
        if len(old_pts) < 8:  # Need at least 8 points for fundamental matrix
            return 0, 0, 0
        
        # Find fundamental matrix
        F, mask = cv2.findFundamentalMat(old_pts, new_pts, cv2.FM_RANSAC, 1.0, 0.99)
        
        if F is None:
            return 0, 0, 0
        
        # Get essential matrix
        E = self.K.T @ F @ self.K
        
        # Decompose essential matrix to get rotation and translation
        _, R, t, mask = cv2.recoverPose(E, old_pts, new_pts, self.K)
        
        # The translation vector t is normalized, we need to scale it
        # For monocular vision, we use ground plane assumption
        
        # Calculate translation magnitude using ground plane constraint
        # Assuming most features are on the ground plane
        scale = self.estimate_scale_from_ground_plane(old_pts, new_pts, R, t)
        
        # Translation in camera frame
        translation = t * scale
        
        # Forward velocity (Z direction in camera frame)
        vz = translation[2, 0]
        
        # Lateral velocity (X direction)
        vx = translation[0, 0]
        
        # Rotation around Y axis (yaw rate)
        yaw_rate = np.arctan2(R[2, 0], R[0, 0])
        
        return vz, vx, yaw_rate
    
    def estimate_scale_from_ground_plane(self, old_pts, new_pts, R, t):
        """
        Estimate scale factor using ground plane assumption
        """
        # For points on the ground plane, we know their height is camera_height
        # This gives us the scale factor
        
        # Simple approach: use median flow magnitude and camera height
        flow_magnitudes = np.linalg.norm(new_pts - old_pts, axis=1)
        median_flow = np.median(flow_magnitudes)
        
        if median_flow < 1e-3:
            return 0
        
        # Approximate scale based on camera height and typical ground feature movement
        # This is a simplified approach - in practice, more sophisticated methods are used
        scale = self.camera_height * 10.0 / median_flow  # Empirical scaling factor
        
        return scale
    
    def pixel_to_normalized(self, points):
        """Convert pixel coordinates to normalized camera coordinates"""
        normalized = np.zeros_like(points)
        normalized[:, 0] = (points[:, 0] - self.cx) / self.fx
        normalized[:, 1] = (points[:, 1] - self.cy) / self.fy
        return normalized
    
    def estimate_velocity_visual_odometry(self, frame, fps=10.0):
        """
        Estimate ego vehicle velocity using visual odometry
        """
        # Detect features
        corners, gray = self.detect_features(frame)
        
        velocity_kmh = 0
        lateral_velocity = 0
        yaw_rate = 0
        
        if self.prev_frame is not None and corners is not None:
            # Calculate optical flow
            old_pts, new_pts, status = self.calculate_optical_flow(
                self.prev_frame['gray'], gray, self.prev_frame['corners']
            )
            
            if old_pts is not None and new_pts is not None and len(old_pts) > 8:
                # Estimate ego motion
                vz, vx, yaw = self.estimate_ego_motion(old_pts, new_pts)
                
                # Convert to velocity (m/s to km/h)
                velocity_kmh = abs(vz) * fps * 3.6
                lateral_velocity = vx * fps * 3.6
                yaw_rate = yaw * fps
                
                # Sanity check - cap unrealistic velocities
                velocity_kmh = min(velocity_kmh, 200)  # Max 200 km/h
        
        # Update previous frame
        self.prev_frame = {
            'gray': gray,
            'corners': corners
        }
        
        return velocity_kmh, lateral_velocity, yaw_rate
    
    def load_ground_truth_velocities(self, oxts_path, frame_count):
        """
        Load ground truth velocities from KITTI oxts data
        """
        gt_velocities = []
        
        for i in range(frame_count):
            oxts_file = os.path.join(oxts_path, f'{i:06d}.txt')
            if os.path.exists(oxts_file):
                with open(oxts_file, 'r') as f:
                    data = f.readline().strip().split()
                    if len(data) >= 11:
                        # KITTI oxts format: vf, vl, vu are indices 8, 9, 10
                        vf = float(data[8])  # forward velocity (m/s)
                        vl = float(data[9])  # leftward velocity (m/s)
                        # Total velocity magnitude
                        velocity = np.sqrt(vf**2 + vl**2) * 3.6  # Convert to km/h
                        gt_velocities.append(velocity)
                    else:
                        gt_velocities.append(0)
            else:
                gt_velocities.append(0)
        
        return gt_velocities
    
    def visualize_results(self, frame, estimated_velocity, gt_velocity, frame_id):
        """
        Visualize optical flow and velocity comparison
        """
        result = frame.copy()
        
        # Draw optical flow vectors if available
        if self.prev_frame is not None and 'corners' in self.prev_frame:
            corners = self.prev_frame['corners']
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(result, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Add velocity text
        cv2.putText(result, f'Estimated: {estimated_velocity:.1f} km/h', 
                   (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, f'Ground Truth: {gt_velocity:.1f} km/h', 
                   (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result, f'Error: {abs(estimated_velocity - gt_velocity):.1f} km/h', 
                   (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Create velocity comparison plot
        self.frame_velocities.append({
            'frame': frame_id,
            'estimated': estimated_velocity,
            'ground_truth': gt_velocity
        })
        
        # Plot last 200 frames
        fig, ax = plt.subplots(figsize=(8, 4))
        
        if len(self.frame_velocities) > 0:
            recent = self.frame_velocities[-200:]
            frames = [v['frame'] for v in recent]
            est = [v['estimated'] for v in recent]
            gt = [v['ground_truth'] for v in recent]
            
            ax.plot(frames, gt, 'r-', label='Ground Truth', linewidth=2)
            ax.plot(frames, est, 'g-', label='Estimated (Visual Odometry)', linewidth=2)
            
            ax.set_xlabel('Frame')
            ax.set_ylabel('Velocity (km/h)')
            ax.set_title('Ego Vehicle Speed Estimation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(50, max(gt + est) * 1.1))
        
        # Convert plot to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        try:
            buf = canvas.buffer_rgba()
            plot_img = np.asarray(buf)
            plot_img = plot_img[:, :, :3]  # Remove alpha
        except AttributeError:
            buf = canvas.tostring_rgb()
            w, h = canvas.get_width_height()
            plot_img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig)
        
        # Resize and combine
        plot_h, plot_w = plot_img.shape[:2]
        video_h, video_w = result.shape[:2]
        plot_img = cv2.resize(plot_img, (video_w, int(video_w * plot_h / plot_w)))
        
        # Stack vertically
        combined = np.vstack([result, plot_img])
        
        return combined

def process_kitti_ego_speed(video_path, calib_path, oxts_path, output_path):
    """
    Process KITTI sequence and estimate ego vehicle speed
    """
    print("Initializing ego vehicle speed estimator...")
    estimator = EgoVehicleSpeedEstimator(calib_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")
    
    # Load ground truth
    gt_velocities = estimator.load_ground_truth_velocities(oxts_path, frame_count)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height + 400))
    
    frame_id = 0
    errors = []
    
    print("Processing frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Estimate ego vehicle speed
        est_velocity, lateral_vel, yaw_rate = estimator.estimate_velocity_visual_odometry(frame, fps)
        
        # Get ground truth
        gt_velocity = gt_velocities[frame_id] if frame_id < len(gt_velocities) else 0
        
        # Calculate error
        error = abs(est_velocity - gt_velocity)
        errors.append(error)
        
        # Visualize
        result_frame = estimator.visualize_results(frame, est_velocity, gt_velocity, frame_id)
        out.write(result_frame)
        
        frame_id += 1
        
        if frame_id % 50 == 0:
            mae = np.mean(errors) if errors else 0
            print(f"Frame {frame_id}/{frame_count} - Current MAE: {mae:.2f} km/h")
    
    # Final results
    mae = np.mean(errors) if errors else 0
    rmse = np.sqrt(np.mean([e**2 for e in errors])) if errors else 0
    
    print(f"\n=== Final Results ===")
    print(f"Mean Absolute Error: {mae:.2f} km/h")
    print(f"Root Mean Square Error: {rmse:.2f} km/h")
    
    # Save results
    results = {
        'mae': mae,
        'rmse': rmse,
        'frame_velocities': estimator.frame_velocities
    }
    
    with open('ego_speed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nOutput saved to: {output_path}")
    print(f"Results saved to: ego_speed_results.json")
    
    # Create summary plot
    plt.figure(figsize=(12, 6))
    frames = [v['frame'] for v in estimator.frame_velocities]
    est = [v['estimated'] for v in estimator.frame_velocities]
    gt = [v['ground_truth'] for v in estimator.frame_velocities]
    
    plt.plot(frames, gt, 'r-', label='Ground Truth', alpha=0.8)
    plt.plot(frames, est, 'g-', label='Estimated (Visual Odometry)', alpha=0.8)
    plt.xlabel('Frame')
    plt.ylabel('Velocity (km/h)')
    plt.title(f'Ego Vehicle Speed Estimation - MAE: {mae:.2f} km/h, RMSE: {rmse:.2f} km/h')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ego_speed_results.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    # KITTI dataset paths
    video_path = "output_0026_image02.mp4"
    calib_path = "2011_09_29_calib/calib_cam_to_cam.txt"
    oxts_path = "2011_09_29_drive_0026_sync/oxts/"
    output_path = "ego_speed_estimation_output.mp4"
    
    process_kitti_ego_speed(video_path, calib_path, oxts_path, output_path)