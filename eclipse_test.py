import cv2
from pupil_detectors import Detector2D
import asyncio
import websockets
import json

clients = set()

async def ws_handler(ws, path):
    clients.add(ws)
    try:
        async for msg in ws:
            pass
    finally:
        clients.remove(ws)

async def broadcast(gx, gy):
    if not clients:
        return
    msg = json.dumps({"gx": gx, "gy": gy})
    await asyncio.gather(*[ws.send(msg) for ws in clients])

def main():
    # --- カメラを開く（HBVCAM） ---
    # カメラが複数ある場合は 0,1,2 を試して正しいものを使ってください
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Could not open camera")
        return

    # 解像度を指定したければここで
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # カメラ情報を確認
    print("Camera opened successfully")
    print(f"Frame size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    detector = Detector2D()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print(f"frame grab failed (attempt {frame_count})")
            if frame_count > 10:  # Try a few times before giving up
                print("Camera is not returning frames. Trying to reset...")
                cap.release()
                import time
                time.sleep(1)
                cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
                frame_count = 0
                if not cap.isOpened():
                    print("Failed to reopen camera")
                    break
            continue

        # frame: BGR (H, W, 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # --- 瞳孔検出 ---
        result = detector.detect(gray)
        ellipse = result["ellipse"]

        if ellipse is not None:
            center = tuple(int(v) for v in ellipse["center"])
            axes   = tuple(int(v / 2) for v in ellipse["axes"])
            angle  = ellipse["angle"]

            # 瞳孔楕円を描画
            cv2.ellipse(
                frame,
                center,
                axes,
                angle,
                0, 360,
                (0, 0, 255),
                2,
            )

            # center → -1〜1 に正規化（後で pan/tilt に変換しやすいように）
            nx = (center[0] / w) * 2.0 - 1.0
            ny = (1.0 - center[1] / h) * 2.0 - 1.0  # 上を+にしたいので反転
            
           # await broadcast(nx, ny)

            conf = result.get("confidence", 1.0)
            print(f"gaze approx: ({nx:.3f}, {ny:.3f}), conf={conf:.3f}")

            # ← ここで nx, ny を WebSocket 経由でブラウザに送れば
            #    そのまま updateGaze(nx, ny) に繋がる

        cv2.imshow("eye", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESCで終了
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
