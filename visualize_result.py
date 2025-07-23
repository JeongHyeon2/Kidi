# visualize_result.py (실제 속도와 추정 속도 동시 표시)

import cv2
import numpy as np
import argparse
from pathlib import Path

def load_results(filepath):
    """결과 파일을 읽어 (필터링된 속도, 실제 속도) 튜플을 저장하는 딕셔너리로 변환합니다."""
    results = {}
    with open(filepath, 'r') as f:
        next(f) # 헤더 스킵
        for line in f:
            # 4개의 값을 모두 읽어옴
            frame, est_speed, filt_speed, gt_speed = line.strip().split(',')
            # 영상에 표시할 값으로 (필터링된 속도, 실제 속도)를 튜플로 저장
            results[int(frame)] = (float(filt_speed), float(gt_speed))
    return results

def main(args):
    print("1. Loading speed results...")
    results = load_results(args.result_file)
    
    print("2. Opening video files...")
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.input_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    
    frame_idx = 0
    print("\n3. Processing video and overlaying speed...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 현재 프레임에 해당하는 (필터링된 속도, 실제 속도) 튜플 가져오기
        filtered_speed, gt_speed = results.get(frame_idx - 1, (-1.0, -1.0))
        
        # --- 텍스트 표시 로직 수정 ---
        # 1. 추정 속도 텍스트
        if filtered_speed >= 0:
            text_est = f"Est. Speed: {filtered_speed:.2f} km/h"
        else:
            text_est = "Est. Speed: N/A"
        
        # 2. 실제 속도 텍스트
        if gt_speed >= 0:
            text_gt = f"GT   Speed: {gt_speed:.2f} km/h"
        else:
            text_gt = "GT   Speed: N/A"
            
        # 텍스트 배경을 위한 검은색 박스 그리기 (두 줄을 표시할 수 있도록 더 길게)
        cv2.rectangle(frame, (10, 10), (420, 90), (0, 0, 0), -1)
        # 추정 속도 텍스트 그리기 (윗줄)
        cv2.putText(frame, text_est, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # 실제 속도 텍스트 그리기 (아랫줄)
        cv2.putText(frame, text_gt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"  - Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"\n✅ Visualization complete. Video saved to {args.output_video}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overlay speed on video.')
    parser.add_argument('--input-video', type=str, default='C:/workspace/kidi/video/output_0004.mp4')
    parser.add_argument('--result-file', type=str, default='C:/workspace/kidi/output/results.txt')
    parser.add_argument('--output-video', type=str, default='C:/workspace/kidi/video/output_0004_result.mp4')
    args = parser.parse_args()
    main(args)