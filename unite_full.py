import threading
import time
import math
import serial
import zmq
import msgpack
import numpy as np
import tkinter as tk
from tkinter import ttk
from serial.tools import list_ports

# =========================================================
# 設定
# =========================================================
PUPIL_REMOTE = "tcp://127.0.0.1:50020"
# SERIAL_PORT = "/dev/cu.usbserial-815E126BF0" #"COM4"
BAUDRATE = 9600

SEND_HZ = 40
MIN_DELTA_DEG = 1.5
SMOOTHING = 0.3

CONFIDENCE_TH = 0.95
FIXATION_STALE_MS = 800

# =========================================================
# グローバル状態（JS state 完全対応）
# =========================================================
running = True

# モード
is_calibrating = False
tracking = False

# 入力
latest_gaze = None            # (x,y)
latest_fix = None             # dict

# キャリブ
calib_points = []             # (x,y,pan,tilt)
model = None                  # (Mp, Mt)

# 手動操作
manual_pan = 90
manual_tilt = 90
manual_z = None               # None = auto

# 出力ターゲット
target_pan = 90
target_tilt = 90
target_z = None

# ===== Serial =====
serial_port = None      # 選択されたポート名
serial_connected = False

# ===== Z UI =====
Z_MIN = 0
Z_MAX = 1000
manual_z = None         # None = auto

def list_serial_ports():
    return [p.device for p in list_ports.comports()]



# =========================================================
# 数学ユーティリティ（Affine）
# =========================================================
def fit_affine(points):
    A, Bp, Bt = [], [], []
    for x, y, p, t in points:
        A.append([x, y, 1])
        Bp.append(p)
        Bt.append(t)
    A = np.array(A)
    Mp, *_ = np.linalg.lstsq(A, np.array(Bp), rcond=None)
    Mt, *_ = np.linalg.lstsq(A, np.array(Bt), rcond=None)
    return Mp, Mt

def predict_affine(model, x, y):
    Mp, Mt = model
    v = np.array([x, y, 1])
    return float(Mp @ v), float(Mt @ v)

def map_z_mm_to_servo(z_mm):
    Z_NEAR = 300
    Z_FAR = 5000
    z = max(Z_NEAR, min(Z_FAR, z_mm))
    return int((z - Z_NEAR) / (Z_FAR - Z_NEAR) * 1000)

# =========================================================
# Pupil 受信スレッド
# =========================================================
def pupil_thread():
    global latest_gaze, latest_fix

    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.connect(PUPIL_REMOTE)
    req.send_string("SUB_PORT")
    sub_port = req.recv_string()

    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{sub_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "gaze.")
    sub.setsockopt_string(zmq.SUBSCRIBE, "fixations")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    while running:
        socks = dict(poller.poll(10))
        if sub in socks:
            topic, payload = sub.recv_multipart()
            msg = msgpack.loads(payload, raw=False)

            if topic.startswith(b"gaze"):
                n = msg.get("norm_pos")
                if n and len(n) == 2:
                    latest_gaze = (float(n[0]), float(n[1]))

            if topic.startswith(b"fixations"):
                if msg.get("method") != "3d gaze":
                    continue
                gp = msg.get("gaze_point_3d")
                n = msg.get("norm_pos")
                if gp and n:
                    latest_fix = {
                        "x": float(n[0]),
                        "y": float(n[1]),
                        "z": float(gp[2]),
                        "t": time.perf_counter(),
                        "confidence": float(msg.get("confidence", 1.0)),
                    }

# =========================================================
# コントローラ（JS useEffect 相当）
# =========================================================
def controller_thread():
    global target_pan, target_tilt, target_z

    last_pan = None
    last_tilt = None

    while running:
        if tracking and model and latest_gaze:
            x, y = latest_gaze
            pan, tilt = predict_affine(model, x, y)

            if last_pan is not None:
                pan = last_pan + SMOOTHING * (pan - last_pan)
                tilt = last_tilt + SMOOTHING * (tilt - last_tilt)

            target_pan = pan
            target_tilt = tilt

            # Z
            if manual_z is not None:
                target_z = manual_z
            elif latest_fix:
                age = (time.perf_counter() - latest_fix["t"]) * 1000
                if age <= FIXATION_STALE_MS and latest_fix["confidence"] >= CONFIDENCE_TH:
                    target_z = map_z_mm_to_servo(latest_fix["z"])
        else:
            target_pan = manual_pan
            target_tilt = manual_tilt
            target_z = manual_z

        last_pan = target_pan
        last_tilt = target_tilt
        time.sleep(0.01)



# =========================================================
# Serial 送信スレッド
# =========================================================
def serial_thread():
    global serial_connected

    ser = None
    last_sent = (None, None, None)
    period = 1.0 / SEND_HZ

    while running:
        if not serial_connected or serial_port is None:
            time.sleep(0.1)
            continue

        if ser is None:
            try:
                print("[SERIAL] opening", serial_port)
                ser = serial.Serial(serial_port, BAUDRATE, timeout=1)
                time.sleep(2)
                ser.write(b"START\r\n")
                ser.flush()
                print("[SERIAL] connected")
            except Exception as e:
                print("[SERIAL] open failed:", e)
                ser = None
                serial_connected = False
                continue

        pan = int(round(target_pan))
        tilt = int(round(target_tilt))
        z = target_z

        if (pan, tilt, z) != last_sent:
            cmd = f"P{pan}T{tilt}"
            if z is not None:
                cmd += f"Z{z}"
            cmd += "\r\n"
            print("[SERIAL SEND]", cmd.strip())


            try:
                ser.write(cmd.encode())
                ser.flush()
                last_sent = (pan, tilt, z)
            except Exception as e:
                print("[SERIAL] write error:", e)
                ser.close()
                ser = None
                serial_connected = False

        time.sleep(period)

    if ser:
        try:
            ser.write(b"STOP\r\n")
            ser.close()
        except:
            pass
        




# =========================================================
# UI（Tkinter）
# =========================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pupil PanTilt Controller")
        self.geometry("380x420")
        
        # === Serial UI ===
        ttk.Label(self, text="Serial Port").pack(pady=(8,0))

        self.port_var = tk.StringVar()
        self.port_menu = ttk.Combobox(
            self, textvariable=self.port_var,
            values=list_serial_ports(),
            state="readonly"
        )
        self.port_menu.pack(fill="x")

        ttk.Button(self, text="Refresh Ports", command=self.refresh_ports).pack(fill="x")
        ttk.Button(self, text="Connect / Disconnect", command=self.toggle_serial).pack(fill="x")

        self.serial_status = ttk.Label(self, text="Serial: DISCONNECTED")
        self.serial_status.pack(pady=4)


        ttk.Button(self, text="キャリブ開始", command=self.start_calib).pack(fill="x")
        ttk.Button(self, text="キャリブ終了", command=self.stop_calib).pack(fill="x")
        ttk.Button(self, text="Tracking ON/OFF", command=self.toggle_tracking).pack(fill="x", pady=4)

        self.pan = tk.Scale(self, from_=0, to=180, orient="horizontal", label="Pan",
                            command=lambda v: self.set_pan(v))
        self.pan.set(90)
        self.pan.pack(fill="x")

        self.tilt = tk.Scale(self, from_=0, to=180, orient="horizontal", label="Tilt",
                             command=lambda v: self.set_tilt(v))
        self.tilt.set(90)
        self.tilt.pack(fill="x")
        
        # === Z UI ===
        ttk.Label(self, text="Z (Linear)").pack(pady=(10,0))

        self.z_scale = tk.Scale(
            self, from_=Z_MIN, to=Z_MAX,
            orient="horizontal",
            command=self.set_z
        )
        self.z_scale.set(500)
        self.z_scale.pack(fill="x")

        ttk.Button(self, text="Z Auto (Fixation)", command=self.set_z_auto).pack(fill="x")


        ttk.Button(self, text="Add Point", command=self.add_point).pack(fill="x", pady=4)
        ttk.Button(self, text="Fit Model", command=self.fit).pack(fill="x")

        self.lbl = ttk.Label(self, text="status")
        self.lbl.pack(pady=6)

        self.after(100, self.update_ui)

    def start_calib(self):
        global is_calibrating, tracking
        is_calibrating = True
        tracking = False
        print("CALIB START")

    def stop_calib(self):
        global is_calibrating
        is_calibrating = False
        print("CALIB END")

    def toggle_tracking(self):
        global tracking, manual_z
        tracking = not tracking

        if tracking:
            manual_z = None   # ← ここが超重要
            print("TRACKING = ON (Z -> auto)")
        else:
            print("TRACKING = OFF")


    def set_pan(self, v):
        global manual_pan
        manual_pan = float(v)

    def set_tilt(self, v):
        global manual_tilt
        manual_tilt = float(v)

    def add_point(self):
        if not latest_gaze:
            print("no gaze")
            return
        calib_points.append((*latest_gaze, manual_pan, manual_tilt))
        print("ADD", calib_points[-1])

    def fit(self):
        global model
        if len(calib_points) >= 3:
            model = fit_affine(calib_points)
            print("MODEL FITTED")

    
    def update_ui(self):
        if latest_gaze:
            self.lbl.config(text=f"gaze {latest_gaze[0]:.3f},{latest_gaze[1]:.3f}")

        if serial_connected:
            self.serial_status.config(text=f"Serial: CONNECTED ({serial_port})")

        self.after(100, self.update_ui)
        
    def refresh_ports(self):
        self.port_menu["values"] = list_serial_ports()

    def toggle_serial(self):
        global serial_connected, serial_port
        if not serial_connected:
            serial_port = self.port_var.get()
            if serial_port:
                serial_connected = True
                self.serial_status.config(text=f"Serial: CONNECTING ({serial_port})")
        else:
            serial_connected = False
            self.serial_status.config(text="Serial: DISCONNECTED")
            
    def set_z(self, v):
        global manual_z
        manual_z = int(v)

    def set_z_auto(self):
        global manual_z
        manual_z = None

# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    threading.Thread(target=pupil_thread, daemon=True).start()
    threading.Thread(target=controller_thread, daemon=True).start()
    threading.Thread(target=serial_thread, daemon=True).start()

    App().mainloop()
    running = False
