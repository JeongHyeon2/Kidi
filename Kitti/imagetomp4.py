import cv2
import os
from pathlib import Path

def images_to_video(image_dir, output_path, fps=10):
    image_files = sorted(Path(image_dir).glob("*.png"))
    if not image_files:
        raise ValueError("이미지를 찾을 수 없습니다")

    # 첫 이미지로 해상도 추출
    frame = cv2.imread(str(image_files[0]))
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        out.write(frame)

    out.release()
    print(f"✅ 영상 저장 완료: {output_path}")

# 실행 예시
image_folder = "./2011_09_29_drive_0004_sync/image_02/data"
output_video = "./output_0004.mp4"
images_to_video(image_folder, output_video)
