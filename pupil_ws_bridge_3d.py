import asyncio
import json
import time

import msgpack
import zmq
import websockets

PUPIL_REMOTE_ADDR = "tcp://127.0.0.1:50020"

WS_HOST = "127.0.0.1"
WS_PORT = 8080

clients = set()


async def ws_handler(websocket):
    print("[WS] client connected")
    clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        clients.discard(websocket)
        print("[WS] client disconnected")


async def broadcast(data: str):
    if not clients:
        return
    await asyncio.gather(*(ws.send(data) for ws in list(clients)), return_exceptions=True)


def connect_pupil_subscriber():
    ctx = zmq.Context.instance()

    req = ctx.socket(zmq.REQ)
    req.connect(PUPIL_REMOTE_ADDR)
    req.send_string("SUB_PORT")
    sub_port = req.recv_string()
    print("[Pupil] SUB_PORT =", sub_port)

    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{sub_port}")

    # gaze 全体を取る（2D/3Dどちらも）
    sub.setsockopt_string(zmq.SUBSCRIBE, "gaze.")
    sub.setsockopt_string(zmq.SUBSCRIBE, "fixation")

    return sub


async def pupil_loop():
    sub = connect_pupil_subscriber()
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    print("[Pupil] listening for gaze.* messages...")

    while True:
        socks = dict(poller.poll(10))
        if sub in socks:
            topic_b, payload = sub.recv_multipart()
            topic = topic_b.decode("utf-8", errors="ignore")

            try:
                msg = msgpack.loads(payload, raw=False)
            except Exception as e:
                print("[Pupil] msgpack error:", e)
                continue

            conf = float(msg.get("confidence", 0.0))

            # ===== 3D gaze =====
            # docs: gaze datum can include eye_center_3d, gaze_point_3d (unit: mm)
            # ===== fixation (3D gaze) =====
            if topic.startswith("fixations"):
                if msg.get("method") != "3d gaze":
                    continue
                
                conf = float(msg.get("confidence", 0.0))
                duration = msg.get("duration", 0.0)

                gaze_point = msg.get("gaze_point_3d")
                #print(msg.get("method"))
                if not gaze_point or len(gaze_point) != 3:
                    continue

                x, y, z = gaze_point  # mm
                twoD = msg.get("norm_pos")
                # print(f"[Pupil] fixation3d: x={x} y={y} z={z} mm, conf={conf}, duration={duration} ms")

                out = {
                    "topic": "fixation3d",
                    "point": [float(x), float(y), float(z)],  # mm
                    "norm_pos": twoD,
                    "confidence": conf,
                    "duration_ms": duration,
                    "t": time.time(),
                }

                await broadcast(json.dumps(out))
                continue

            # ===== gaze normal 3D (raw) =====

            # gaze_normal = msg.get("gaze_normal_3d")
            # # print("gaze_normal:", gaze_normal)
            # if gaze_normal and len(gaze_normal) == 3:
                
            #     nx, ny, nz = map(float, gaze_normal)

            #     conf = float(msg.get("confidence", 0.0))
            #     #norm_pos = msg.get("norm_pos")

            #     out = {
            #         "topic": "gaze_normal_3d",
            #         "normal": [nx, ny, nz],   # unit vector (raw)
            #         #"norm_pos": norm_pos,     # optional
            #         "confidence": conf,
            #         "t": time.time(),
            #     }

            #     await broadcast(json.dumps(out))
            #     continue



            # ===== 2D gaze =====
            norm_pos = msg.get("norm_pos")
            if norm_pos and len(norm_pos) == 2:
                nx, ny = norm_pos
                # Pupil: left-bottom origin -> left-top origin
                x = float(nx)
                y = float(ny)  # 1.0 - 
                # print("norm_pos:", norm_pos)

                out = {
                    "topic": "lookat",
                    "x": x,
                    "y": y,
                    "confidence": conf,
                    "t": time.time(),
                }
                await broadcast(json.dumps(out))

        await asyncio.sleep(0.001)


async def main():
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        print(f"[WS] listening on ws://{WS_HOST}:{WS_PORT}")
        await pupil_loop()


if __name__ == "__main__":
    asyncio.run(main())
