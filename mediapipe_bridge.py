#!/usr/bin/env python3
import asyncio
import json

import cv2
import websockets
import mediapipe as mp

# ==== MediaPipe FaceMesh 設定 ====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,          # 虹彩ランドマークを含める
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 左目の虹彩ランドマーク（MediaPipe 固定インデックス）
# 468, 469, 470, 471 あたりが左目虹彩
LEFT_IRIS_IDXS = [468, 469, 470, 471]
RIGHT_IRIS_IDXS = [473, 474, 475, 476]


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

    loop = asyncio.get_running_loop()
    last_good = None  # (gx, gy, conf)

    try:
        while True:
            # OpenCV はブロッキングなのでスレッドへ
            ret, frame = await loop.run_in_executor(None, cap.read)

            if not ret:
                print("Frame grab failed")
                # フレーム取得に失敗したときは直近の良い値を再送
                if last_good is not None:
                    gx, gy, conf = last_good
                    msg = json.dumps({"gx": gx, "gy": gy, "confidence": conf})
                    await ws.send(msg)
                await asyncio.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # MediaPipe は RGB 想定
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            iris_cx = None
            iris_cy = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                xs = []
                ys = []
                for idx in RIGHT_IRIS_IDXS:
                    lm = face_landmarks.landmark[idx]
                    xs.append(lm.x)
                    ys.append(lm.y)

                if xs and ys:
                    # 画像全体に対する正規化座標 (0〜1)
                    iris_cx = sum(xs) / len(xs)
                    iris_cy = sum(ys) / len(ys)

            if iris_cx is not None and iris_cy is not None:
                # デバッグ用にポイントを描画
                cx_pix = int(iris_cx * w)
                cy_pix = int(iris_cy * h)
                cv2.circle(frame, (cx_pix, cy_pix), 5, (0, 0, 255), -1)

                # 0〜1 -> -1〜1 に変換（元コードに合わせる）
                gx = iris_cx * 2.0 - 1.0
                gy = (1.0 - iris_cy) * 2.0 - 1.0

                conf = 1.0  # MediaPipe からはピクセル毎の信頼度がないので「検出できたら1.0」とする

                last_good = (gx, gy, conf)
                msg = json.dumps({"gx": gx, "gy": gy, "confidence": conf})
                await ws.send(msg)
                print(f"gx={gx:.3f}, gy={gy:.3f}, conf={conf:.3f}")

            else:
                # 虹彩が取れない場合は last_good を再送
                if last_good is not None:
                    gx, gy, conf = last_good
                    msg = json.dumps({"gx": gx, "gy": gy, "confidence": conf})
                    await ws.send(msg)

            cv2.imshow("eye (mediapipe)", frame)
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
