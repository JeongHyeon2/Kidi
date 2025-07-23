# read_gt_speed.py

import numpy as np
import argparse
from pathlib import Path

def main(args):
    oxts_dir = Path(args.sequence_path) / 'oxts' / 'data'
    oxts_files = sorted(list(oxts_dir.glob('*.txt')))
    
    print(f"Reading ground truth speed from {len(oxts_files)} files...")
    
    with open(args.gt_output_file, 'w') as f:
        f.write("frame,speed_kmh\n")
        
        for i, oxts_file in enumerate(oxts_files):
            with open(oxts_file, 'r') as oxts_f:
                # 8, 9, 10번째 값이 각각 전방, 좌측, 상방 속도(m/s)
                data = np.fromstring(oxts_f.readline(), dtype=np.float64, sep=' ')
                speed_mps = np.linalg.norm(data[8:11]) # 3D 속도 벡터의 크기
                speed_kmh = speed_mps * 3.6
                
                f.write(f"{i},{speed_kmh}\n")
    
    print(f"✅ Ground truth speed saved to {args.gt_output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read KITTI ground truth speed.')
    parser.add_argument('--sequence_path', type=str, 
                        default='C:/workspace/kidi/Kitti/2011_09_29_drive_0004_sync')
    parser.add_argument('--gt_output_file', type=str, 
                        default='C:/workspace/kidi/output/gt_speed.txt')
    args = parser.parse_args()
    main(args)