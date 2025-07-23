# cal.py (최종 + 아웃라이어 제거 및 이동 평균 필터)

import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from collections import deque

# 다른 함수들은 모두 동일...
def parse_kitti_calib(filepath):
    if not filepath.exists(): raise FileNotFoundError(f"캘리브레이션 파일을 찾을 수 없습니다: {filepath}")
    data = {};
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            try: data[key] = np.array([float(x) for x in value.split()])
            except ValueError: pass
    p_rect_02 = data['P_rect_02'].reshape(3, 4)
    fx, fy = p_rect_02[0, 0], p_rect_02[1, 1]
    cx, cy = p_rect_02[0, 2], p_rect_02[1, 2]
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

def load_kitti_timestamps(filepath):
    if not filepath.exists(): raise FileNotFoundError(f"타임스탬프 파일을 찾을 수 없습니다: {filepath}")
    timestamps = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            dt_object = datetime.strptime(line.strip()[:-3], '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(dt_object.timestamp())
    timestamps = np.array(timestamps)
    return np.diff(timestamps)

def load_gt_speeds(filepath):
    if not filepath.exists(): raise FileNotFoundError(f"실제 속도 파일을 찾을 수 없습니다: {filepath}")
    gt_speeds = np.loadtxt(filepath, delimiter=',', skiprows=1)
    return gt_speeds[:, 1]

def solve_ego_motion_ransac(points, flow_vectors, intrinsics, iterations=100, threshold=1.0):
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    best_inliers_count = -1; best_T = None
    num_points = points.shape[0]
    if num_points < 3: return None
    A = np.zeros((num_points * 2, 3)); b = np.zeros(num_points * 2)
    x_coords, y_coords, Z = points[:, 0], points[:, 1], points[:, 2]
    u, v = flow_vectors[:, 0], flow_vectors[:, 1]
    A[0::2, 0] = -fx / Z; A[0::2, 2] = (x_coords - cx) / Z
    A[1::2, 1] = -fy / Z; A[1::2, 2] = (y_coords - cy) / Z
    b[0::2] = u; b[1::2] = v
    for i in range(iterations):
        sample_indices = np.random.choice(num_points, 3, replace=False)
        A_sample = np.vstack([A[2*j:2*j+2] for j in sample_indices])
        b_sample = np.hstack([b[2*j:2*j+2] for j in sample_indices])
        try: T_sample, _, _, _ = np.linalg.lstsq(A_sample, b_sample, rcond=None)
        except np.linalg.LinAlgError: continue
        residuals = np.abs(b - A @ T_sample)
        pixel_errors = np.sqrt(residuals[0::2]**2 + residuals[1::2]**2)
        inliers_mask = pixel_errors < threshold
        inliers_count = np.sum(inliers_mask)
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_T = T_sample
    if best_inliers_count > 3:
        inlier_indices = np.where(inliers_mask)[0]
        A_inliers = np.vstack([A[2*j:2*j+2] for j in inlier_indices])
        b_inliers = np.hstack([b[2*j:2*j+2] for j in inlier_indices])
        try:
            final_T, _, _, _ = np.linalg.lstsq(A_inliers, b_inliers, rcond=None)
            return final_T
        except np.linalg.LinAlgError: return best_T
    return best_T


def main(args):
    data_dir = Path(args.data_dir); calib_file = Path(args.kitti_dir) / args.calib_name / 'calib_cam_to_cam.txt'
    flow_dir, depth_dir, mask_dir = data_dir / 'flow', data_dir / 'depth', data_dir / 'mask'
    ts_file = Path(args.kitti_dir) / args.sequence_name / 'image_02' / 'timestamps.txt'
    
    print("1. Loading data...")
    intrinsics = parse_kitti_calib(calib_file)
    time_diffs = load_kitti_timestamps(ts_file)
    gt_speeds = load_gt_speeds(Path(args.gt_file))
    
    print("\n2. Starting Robust Ego-Motion & Speed Calculation with Outlier-Rejected Filtering...")
    flow_files = sorted(list(flow_dir.glob('*.npy')))
    
    SCALE_FACTOR = 13.8018
    
    # --- 필터 설정 ---
    WINDOW_SIZE = 10 # 평균 낼 프레임 개수 (조절 가능)
    JUMP_THRESHOLD = 10.0 # 이전 프레임과 이 값(km/h) 이상 차이나면 아웃라이어로 간주
    speed_window = deque(maxlen=WINDOW_SIZE)
    filtered_speed_kmh = -1.0
    
    with open(args.result_file, 'w') as f:
        f.write("frame,estimated_speed_kmh,filtered_speed_kmh,ground_truth_speed_kmh\n")
        
        for i, flow_path in enumerate(flow_files):
            frame_idx = int(flow_path.stem)
            depth_path = depth_dir / f'{frame_idx:010d}.npy'; mask_path = mask_dir / f'{frame_idx:010d}.npy'
            
            if not (depth_path.exists() and mask_path.exists()): continue
            
            depth = np.load(depth_path) * SCALE_FACTOR; flow = np.load(flow_path); mask = np.load(mask_path)
            height, width = depth.shape
            mask[:int(height * 0.4), :] = 0; mask[depth > 80] = 0
            y_coords, x_coords = np.where(mask == 255)
            points_3d = np.vstack((x_coords, y_coords, depth[y_coords, x_coords])).T
            flow_vectors_2d = flow[y_coords, x_coords]
            
            T = solve_ego_motion_ransac(points_3d, flow_vectors_2d, intrinsics)
            
            estimated_speed_kmh = -1.0
            if T is not None:
                delta_t = time_diffs[frame_idx]
                speed_mps = np.linalg.norm(T) / delta_t
                estimated_speed_kmh = speed_mps * 3.6

            # --- 아웃라이어 제거 + 이동 평균 필터링 로직 ---
            is_outlier = False
            if estimated_speed_kmh >= 0:
                # 필터가 어느 정도 안정화된 후에만 아웃라이어 체크
                if len(speed_window) > WINDOW_SIZE / 2 and filtered_speed_kmh > 0:
                    if abs(estimated_speed_kmh - filtered_speed_kmh) > JUMP_THRESHOLD:
                        is_outlier = True # 튀는 값이므로 무시
                
                if not is_outlier:
                    speed_window.append(estimated_speed_kmh)
            
            if len(speed_window) > 0:
                filtered_speed_kmh = np.mean(speed_window) # 다시 mean 사용

            gt_speed_kmh = gt_speeds[frame_idx] if frame_idx < len(gt_speeds) else -1.0
            
            outlier_char = "⚠️" if is_outlier else "🚗"
            print(f"  Frame {frame_idx:04d} | {outlier_char} Filt. Speed: {filtered_speed_kmh:6.2f} km/h | ✅ GT Speed: {gt_speed_kmh:6.2f} km/h")
            
            f.write(f"{frame_idx},{estimated_speed_kmh},{filtered_speed_kmh},{gt_speed_kmh}\n")
    
    print(f"\n✅ Calculation finished. Results saved to {args.result_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate and Compare vehicle speed robustly with temporal filtering.')
    parser.add_argument('--kitti_dir', type=str, default='C:/workspace/kidi/Kitti')
    parser.add_argument('--sequence_name', type=str, default='2011_09_29_drive_0004_sync')
    parser.add_argument('--calib_name', type=str, default='2011_09_29_calib')
    parser.add_argument('--data_dir', type=str, default='C:/workspace/kidi/output/2011_09_29_drive_0004_sync')
    parser.add_argument('--result_file', type=str, default='C:/workspace/kidi/output/results.txt')
    parser.add_argument('--gt_file', type=str, default='C:/workspace/kidi/output/gt_speed.txt')
    
    args = parser.parse_args()
    main(args)