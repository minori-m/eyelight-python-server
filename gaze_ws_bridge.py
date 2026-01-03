#!/usr/bin/env python3
import asyncio
import json

import cv2
import websockets
from pupil_detectors import Detector2D


async def stream_gaze(ws):
    """
    1クライアント専用:
    接続が来たらカメラを開いて、切れるまでずっとgazeを送り続ける。
    """
    print("Client connected:", ws.remote_address)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Could not open camera")
        await ws.close()
        return

    detector = Detector2D()
    loop = asyncio.get_running_loop()
    last_good = None  # (gx, gy, conf)

    try:
        while True:
            # OpenCV はブロッキングなのでスレッドへ
            ret, frame = await loop.run_in_executor(None, cap.read)

            if not ret:
                # フレーム取得に失敗したときは直近の良い値を再送
                print("Frame grab failed")
                if last_good is not None:
                    gx, gy, conf = last_good
                    msg = json.dumps({"gx": gx, "gy": gy, "confidence": conf})
                    await ws.send(msg)
                await asyncio.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]

            result = detector.detect(gray)
            ellipse = result["ellipse"]

            if ellipse is not None:
                center = tuple(int(v) for v in ellipse["center"])
                axes = tuple(int(v / 2) for v in ellipse["axes"])
                angle = ellipse["angle"]

                # 画面デバッグ用の楕円
                cv2.ellipse(frame, center, axes, angle, 0, 360, (0, 0, 255), 2)

                gx = (center[0] / w) * 2.0 - 1.0
                gy = (1.0 - center[1] / h) * 2.0 - 1.0
                conf = float(result.get("confidence", 1.0))

                # ある程度 confidence が高いものだけ「良い値」として採用
                if conf > 0.2:
                    last_good = (gx, gy, conf)
                    msg = json.dumps({"gx": gx, "gy": gy, "confidence": conf})
                    await ws.send(msg)
                    print(f"gx={gx:.3f}, gy={gy:.3f}, conf={conf:.3f}")
                else:
                    # 信頼度が低い場合は last_good を再送
                    if last_good is not None:
                        gx, gy, conf = last_good
                        msg = json.dumps({"gx": gx, "gy": gy, "confidence": conf})
                        await ws.send(msg)

            else:
                # 瞳孔が取れない場合も last_good を再送
                if last_good is not None:
                    gx, gy, conf = last_good
                    msg = json.dumps({"gx": gx, "gy": gy, "confidence": conf})
                    await ws.send(msg)

            cv2.imshow("eye", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except websockets.ConnectionClosed:
        print("Client disconnected:", ws.remote_address)
    finally:
        cap.release()
        cv2.destroyAllWindows()


async def main():
    async with websockets.serve(stream_gaze, "0.0.0.0", 8765):
        print("WebSocket server running at ws://localhost:8765")
        await asyncio.Future()  # サーバーをずっと走らせる


if __name__ == "__main__":
    asyncio.run(main())
