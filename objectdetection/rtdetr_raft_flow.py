import cv2
import numpy as np
import torch
import sys
import argparse
from pathlib import Path
from ultralytics import RTDETR
from collections import defaultdict

# RAFT 경로 추가
sys.path.append('RAFT')
sys.path.append('RAFT/core')

try:
    from raft import RAFT
    from utils import flow_viz
    from utils.utils import InputPadder
    RAFT_AVAILABLE = True
except ImportError:
    print("RAFT not found. Please install RAFT first.")
    print("git clone https://github.com/princeton-vl/RAFT.git")
    RAFT_AVAILABLE = False

# --- 설정 ---
TARGET_CLASS_IDS = [2, 5, 7]  # car, bus, truck
CLASS_NAMES = {2: 'car', 5: 'bus', 7: 'truck'}

# 색상 설정 (BGR)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (255, 0, 128), (0, 128, 255), (0, 255, 128)
]

class SimpleTracker:
    def __init__(self, max_lost=30):
        self.tracks = {}
        self.track_id_count = 0
        self.max_lost = max_lost
        
    def update(self, detections):
        active_tracks = []
        
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['lost'] += 1
                if self.tracks[track_id]['lost'] > self.max_lost:
                    del self.tracks[track_id]
            return active_tracks
        
        used_detections = set()
        
        for track_id, track in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(detections):
                if det_idx in used_detections:
                    continue
                    
                iou = self._calculate_iou(track['bbox'], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_iou > 0.3:
                det = detections[best_det_idx]
                self.tracks[track_id]['bbox'] = det[:4]
                self.tracks[track_id]['conf'] = det[4]
                self.tracks[track_id]['class_id'] = det[5]
                self.tracks[track_id]['lost'] = 0
                used_detections.add(best_det_idx)
                
                active_tracks.append({
                    'track_id': track_id,
                    'bbox': det[:4],
                    'conf': det[4],
                    'class_id': det[5]
                })
            else:
                self.tracks[track_id]['lost'] += 1
                if self.tracks[track_id]['lost'] > self.max_lost:
                    del self.tracks[track_id]
        
        for det_idx, det in enumerate(detections):
            if det_idx not in used_detections:
                self.tracks[self.track_id_count] = {
                    'bbox': det[:4],
                    'conf': det[4],
                    'class_id': det[5],
                    'lost': 0
                }
                active_tracks.append({
                    'track_id': self.track_id_count,
                    'bbox': det[:4],
                    'conf': det[4],
                    'class_id': det[5]
                })
                self.track_id_count += 1
        
        return active_tracks
    
    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0


class VehicleTrackerWithFlow:
    def __init__(self, rtdetr_model="rtdetr-l.pt", raft_model="RAFT/models/raft-things.pth", device='cuda'):
        """
        차량 추적 + 배경 optical flow
        
        Args:
            rtdetr_model: RT-DETR 모델 경로
            raft_model: RAFT 모델 경로
            device: 실행 디바이스
        """
        self.device = device
        
        # RT-DETR 모델 로드
        print(f"Loading RT-DETR model: {rtdetr_model}")
        self.detector = RTDETR(rtdetr_model)
        self.detector.fuse()
        
        # RAFT 모델 로드
        if RAFT_AVAILABLE and Path(raft_model).exists():
            print(f"Loading RAFT model: {raft_model}")
            args = argparse.Namespace()
            args.model = raft_model
            args.small = False
            args.mixed_precision = False
            
            self.flow_model = RAFT(args)
            self.flow_model = torch.nn.DataParallel(self.flow_model)
            self.flow_model.load_state_dict(torch.load(args.model))
            self.flow_model = self.flow_model.module
            self.flow_model.to(device)
            self.flow_model.eval()
            self.use_raft = True
        else:
            print("RAFT model not found. Using OpenCV Farneback instead.")
            self.flow_model = None
            self.use_raft = False
        
        # 트래커 초기화
        self.tracker = SimpleTracker(max_lost=30)
        
        # 통계
        self.frame_count = 0
        self.unique_ids = set()
        self.prev_frame = None
        
    def create_mask(self, frame_shape, tracks, dilation_size=20):
        """추적된 차량들에 대한 마스크 생성"""
        mask = np.ones(frame_shape[:2], dtype=np.uint8) * 255
        
        for track in tracks:
            x1, y1, x2, y2 = [int(x) for x in track['bbox']]
            x1 = max(0, x1 - dilation_size)
            y1 = max(0, y1 - dilation_size)
            x2 = min(frame_shape[1], x2 + dilation_size)
            y2 = min(frame_shape[0], y2 + dilation_size)
            
            mask[y1:y2, x1:x2] = 0
            
        return mask
    
    def compute_flow_raft(self, frame1, frame2):
        """RAFT를 사용하여 optical flow 계산"""
        # RGB로 변환
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # RAFT 입력 형식으로 변환
        image1 = torch.from_numpy(frame1_rgb).permute(2, 0, 1).float()
        image2 = torch.from_numpy(frame2_rgb).permute(2, 0, 1).float()
        
        image1 = image1[None].to(self.device)
        image2 = image2[None].to(self.device)
        
        # 패딩
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        
        # Flow 계산
        with torch.no_grad():
            _, flow_up = self.flow_model(image1, image2, iters=20, test_mode=True)
        
        # 패딩 제거
        flow = padder.unpad(flow_up[0]).cpu().numpy()
        flow = flow.transpose(1, 2, 0)
        
        return flow
    
    def compute_flow_farneback(self, frame1, frame2):
        """OpenCV Farneback optical flow 계산"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        return flow
    
    def visualize_flow(self, flow, mask):
        """Optical flow 시각화 (마스크 적용)"""
        # 마스크 적용
        flow_masked = flow.copy()
        flow_masked[mask == 0] = 0
        
        # Flow 크기 계산
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        # Flow 시각화
        if self.use_raft and 'flow_viz' in sys.modules:
            flow_img = flow_viz.flow_to_image(flow_masked)
        else:
            # HSV 시각화
            h, w = flow.shape[:2]
            fx, fy = flow_masked[:, :, 0], flow_masked[:, :, 1]
            
            ang = np.arctan2(fy, fx) + np.pi
            v = np.sqrt(fx*fx + fy*fy)
            
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 0] = ang * (180 / np.pi / 2)
            hsv[..., 1] = 255
            hsv[..., 2] = np.minimum(v * 4, 255)
            
            flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 마스크 영역을 회색으로 표시
        flow_img[mask == 0] = [64, 64, 64]
        
        return flow_img, flow_magnitude
    
    def process_frame(self, frame):
        """단일 프레임 처리"""
        self.frame_count += 1
        
        # RT-DETR로 객체 탐지
        results = self.detector(frame)[0]
        
        # 차량 클래스만 필터링
        detections = []
        if results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                if class_id in TARGET_CLASS_IDS:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    detections.append((x1, y1, x2, y2, conf, class_id))
        
        # 추적 업데이트
        tracks = self.tracker.update(detections)
        
        # 추적 ID 통계
        for track in tracks:
            self.unique_ids.add(track['track_id'])
        
        # 차량 마스크 생성
        mask = self.create_mask(frame.shape, tracks)
        
        # Optical flow 계산
        flow_img = None
        flow_magnitude = None
        avg_flow = 0
        max_flow = 0
        
        if self.prev_frame is not None:
            if self.use_raft:
                flow = self.compute_flow_raft(self.prev_frame, frame)
            else:
                flow = self.compute_flow_farneback(self.prev_frame, frame)
            
            flow_img, flow_magnitude = self.visualize_flow(flow, mask)
            
            # 배경 영역의 평균 flow 계산
            background_flow = flow_magnitude[mask == 255]
            if len(background_flow) > 0:
                avg_flow = np.mean(background_flow)
                max_flow = np.max(background_flow)
        
        # 현재 프레임 저장
        self.prev_frame = frame.copy()
        
        # 프레임에 바운딩 박스 그리기
        annotated_frame = frame.copy()
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = [int(x) for x in track['bbox']]
            conf = track['conf']
            class_id = track['class_id']
            class_name = CLASS_NAMES.get(class_id, 'vehicle')
            
            color = COLORS[track_id % len(COLORS)]
            
            # 바운딩 박스
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # 라벨
            label = f"#{track_id} {class_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1)
        
        return annotated_frame, flow_img, mask, avg_flow, max_flow
    
    def process_video(self, input_path, output_path, show_flow=True):
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # 출력 크기 결정 (원본 크기로 고정)
        out_width = width
        
        # 비디오 writer 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 처리
            _, flow_img, _, avg_flow, max_flow = self.process_frame(frame)
            
            # 결과 합치기 (flow_img만 사용하도록 수정)
            if flow_img is not None:
                combined = flow_img
                # Flow 이미지에 정보 추가
                method = "RAFT" if self.use_raft else "Farneback"
                info_text = f"Frame: {self.frame_count}/{total_frames}"
                flow_text = f"Avg BG Flow: {avg_flow:.2f} | Max: {max_flow:.2f}"
                
                cv2.putText(combined, f"Optical Flow ({method}) - Background Only", (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(combined, "Gray = Masked Vehicles", (15, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(combined, info_text, (15, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(combined, flow_text, (15, 105), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # 첫 프레임처럼 flow_img가 없는 경우, 검은 화면으로 대체
                combined = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 프레임 저장
            out.write(combined)
            
            # 진행 상황 출력
            if self.frame_count % 30 == 0:
                print(f"Processed {self.frame_count}/{total_frames} frames ({self.frame_count/total_frames*100:.1f}%)")
        
        # 정리
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
        print(f"\n✅ 완료: {output_path}")
        print(f"📊 총 프레임: {self.frame_count}")
        print(f"📊 총 추적된 차량: {len(self.unique_ids)}")


def main():
    parser = argparse.ArgumentParser(description='RT-DETR + RAFT Background Optical Flow')
    parser.add_argument('--input', type=str, default='../video/sample1.mp4', 
                       help='Input video path')
    parser.add_argument('--output', type=str, default='../video/sample1_flow_out.mp4', 
                       help='Output video path')
    parser.add_argument('--rtdetr-model', type=str, default='rtdetr-l.pt', 
                       help='RT-DETR model path')
    parser.add_argument('--raft-model', type=str, default='RAFT/models/raft-things.pth', 
                       help='RAFT model path')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device (cuda or cpu)')
    parser.add_argument('--no-flow', action='store_true', 
                       help='Disable flow visualization')
    
    args = parser.parse_args()
    
    # GPU 확인
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.device = 'cpu'
    
    # 추적기 초기화
    tracker = VehicleTrackerWithFlow(
        rtdetr_model=args.rtdetr_model,
        raft_model=args.raft_model,
        device=args.device
    )
    
    # 비디오 처리
    tracker.process_video(
        input_path=args.input,
        output_path=args.output,
        show_flow=not args.no_flow
    )


if __name__ == "__main__":
    main()