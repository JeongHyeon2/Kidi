# generate_flow.py

import cv2
import numpy as np
import torch
import sys
import argparse
from pathlib import Path
from ultralytics import RTDETR

# RAFT 경로 추가
sys.path.append('RAFT')
sys.path.append('RAFT/core')
from raft import RAFT
from utils.utils import InputPadder

TARGET_CLASS_IDS = [2, 5, 7]  # car, bus, truck

class FlowGenerator:
    def __init__(self, args):
        self.device = args.device
        print("🚀 Initializing RAFT and RT-DETR models...")
        self.detector = RTDETR(args.rtdetr_model)
        self.detector.fuse()
        self.flow_model = self._init_raft(args.raft_model)
        print("✅ Models loaded.")

    def _init_raft(self, model_path):
        raft_args = argparse.Namespace(model=model_path, small=False, mixed_precision=False)
        model = RAFT(raft_args)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(raft_args.model, map_location=self.device, weights_only=True))
        model = model.module.to(self.device)
        model.eval()
        return model

    def create_mask(self, frame_shape, detections):
        mask = np.ones(frame_shape, dtype=np.uint8) * 255
        if detections.boxes is not None:
            for box in detections.boxes:
                if int(box.cls) in TARGET_CLASS_IDS:
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                    cv2.rectangle(mask, (x1-10, y1-10), (x2+10, y2+10), 0, -1)
        return mask

    def run(self, image_dir, output_dir):
        image_files = sorted(list(Path(image_dir).glob('*.png')))
        (output_dir / 'flow').mkdir(parents=True, exist_ok=True)
        (output_dir / 'mask').mkdir(parents=True, exist_ok=True)
        
        prev_frame = None
        for i, img_path in enumerate(image_files):
            frame = cv2.imread(str(img_path))
            if frame is None: continue

            detections = self.detector(frame, verbose=False)[0]
            mask = self.create_mask(frame.shape[:2], detections)
            np.save(output_dir / 'mask' / f'{i:010d}.npy', mask)

            if prev_frame is not None:
                # --- Optical Flow Calculation ---
                frame1_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
                frame2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image1 = torch.from_numpy(frame1_rgb).permute(2, 0, 1).float()[None].to(self.device)
                image2 = torch.from_numpy(frame2_rgb).permute(2, 0, 1).float()[None].to(self.device)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                with torch.no_grad():
                    _, flow_up = self.flow_model(image1, image2, iters=20, test_mode=True)
                flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
                np.save(output_dir / 'flow' / f'{i-1:010d}.npy', flow)
                
            prev_frame = frame.copy()
            print(f"Processed Frame {i+1}/{len(image_files)} for Flow & Mask.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Optical Flow and Masks.')
    
    # --- 모든 경로를 절대 경로로 수정 ---
    parser.add_argument('--rtdetr-model', type=str, 
                        default='C:/workspace/kidi/objectdetection/rtdetr-l.pt')
                        
    parser.add_argument('--raft-model', type=str, 
                        default='C:/workspace/kidi/objectdetection/RAFT/models/raft-things.pth')
                        
    parser.add_argument('--sequence_path', type=str, 
                        default='C:/workspace/kidi/Kitti/2011_09_29_drive_0004_sync')
                        
    parser.add_argument('--output_dir', type=str, 
                        default='C:/workspace/kidi/output/2011_09_29_drive_0004_sync')
                        
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()

    # sequence_path에서 이미지 데이터 경로를 조합
    image_dir = Path(args.sequence_path) / 'image_02' / 'data'
    
    # 출력 경로를 Path 객체로 변환
    output_dir = Path(args.output_dir)

    # FlowGenerator 클래스 실행
    generator = FlowGenerator(args)
    generator.run(image_dir, output_dir)