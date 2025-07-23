# find_scale.py

import numpy as np
import argparse

def main(args):
    # 계산된 속도와 정답 속도 로드
    our_results = np.loadtxt(args.result_file, delimiter=',', skiprows=1)
    gt_results = np.loadtxt(args.gt_file, delimiter=',', skiprows=1)
    
    # 두 데이터의 길이를 짧은 쪽에 맞춤
    min_len = min(len(our_results), len(gt_results))
    our_speeds = our_results[:min_len, 1]
    gt_speeds = gt_results[:min_len, 1]
    
    # 유효한 데이터만 필터링 (계산 실패(-1)나 0인 경우 제외)
    valid_mask = (our_speeds > 0) & (gt_speeds > 0)
    
    # 최적의 스케일 팩터 계산 (정답 속도의 합 / 우리 계산 속도의 합)
    scale_factor = np.sum(gt_speeds[valid_mask]) / np.sum(our_speeds[valid_mask])
    
    print(f"✅ 최적의 뎁스 스케일 팩터 (s) = {scale_factor:.4f}")
    print("\n이제 이 값을 cal.py의 SCALE_FACTOR에 넣고 다시 실행하면 됩니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find optimal scale factor.')
    parser.add_argument('--result_file', type=str, default='C:/workspace/kidi/output/results.txt')
    parser.add_argument('--gt_file', type=str, default='C:/workspace/kidi/output/gt_speed.txt')
    args = parser.parse_args()
    main(args)