import asyncio
import cv2
import numpy as np
import json  # JSON 라이브러리 추가
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

app = FastAPI()
model = YOLO('yolov8n.pt')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("HMD 클라이언트가 연결되었습니다.")

    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                # YOLO 객체 추적 수행
                results = model.track(img, persist=True, verbose=False)

                # --- 여기부터 변경 ---
                
                # 1. 탐지된 객체 정보를 리스트에 저장
                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        # 클래스 ID와 신뢰도 추출
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # 정규화된 바운딩 박스 좌표(xywh) 추출
                        # (x_center, y_center, width, height) 형식, 0~1 사이의 값
                        xywhn = box.xywhn[0].tolist()

                        detections.append({
                            "class_id": class_id,
                            "class_name": model.names[class_id],
                            "confidence": confidence,
                            "box_xywhn": xywhn # 정규화된 좌표 전송
                        })
                
                # 2. 파이썬 딕셔너리를 JSON 문자열로 변환
                json_data = json.dumps({"detections": detections})

                # 3. JSON 데이터를 클라이언트(Unity)로 전송
                await websocket.send_text(json_data)
                
                # (선택사항) 서버에서도 확인하고 싶다면 아래 주석을 해제
                # annotated_frame = results[0].plot()
                # cv2.imshow("Server View", annotated_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            
    except WebSocketDisconnect:
        print("HMD 클라이언트 연결이 끊어졌습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        print("스트리밍을 종료합니다.")
        cv2.destroyAllWindows()