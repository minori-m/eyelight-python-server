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


# ===== WebSocket ハンドラ =====
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
    await asyncio.gather(
        *(ws.send(data) for ws in list(clients)),
        return_exceptions=True,
    )


# ===== Pupil への接続 =====
def connect_pupil_subscriber():
    ctx = zmq.Context.instance()

    req = ctx.socket(zmq.REQ)
    req.connect(PUPIL_REMOTE_ADDR)
    req.send_string("SUB_PORT")
    sub_port = req.recv_string()
    print("[Pupil] SUB_PORT =", sub_port)

    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{sub_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "gaze.")

    return sub


# ===== メインループ =====
async def pupil_loop():
    sub = connect_pupil_subscriber()
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    print("[Pupil] listening for gaze.* messages...")

    while True:
        socks = dict(poller.poll(10))
        if sub in socks:
            topic, payload = sub.recv_multipart()
            try:
                msg = msgpack.loads(payload, raw=False)
            except Exception as e:
                print("[Pupil] msgpack error:", e)
                continue
            
            print(msg)


            norm_pos = msg.get("norm_pos")
            if not norm_pos or len(norm_pos) != 2:
                continue

            conf = float(msg.get("confidence", 0.0))

            nx, ny = norm_pos

            # Pupil は左下原点 → 左上原点へ
            x = float(nx)
            y = 1.0 - float(ny)

            out = {
                "topic": "lookat",
                "x": x,
                "y": y,
                "confidence": conf,
                "t": time.time(),
            }

            await broadcast(json.dumps(out))

        await asyncio.sleep(0.001)


# ===== エントリーポイント =====
async def main():
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        print(f"[WS] listening on ws://{WS_HOST}:{WS_PORT}")
        await pupil_loop()


if __name__ == "__main__":
    asyncio.run(main())
