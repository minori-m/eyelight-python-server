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
import json
import os
from pythonosc import dispatcher, osc_server

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
GAZE_CONF_TH = 0.85   # ← まずはこれおすすめ

Z_PITCH_EPS = 0.1
last_pitch = None

IPHONE_HOLD_SEC = 10.0   # ← ここ重要（0.8〜1.5おすすめ）
iphone_mode_until = 0.0

PAN_MIN, PAN_MAX = 2, 130
TILT_MIN, TILT_MAX = 10, 110

# =========================================================
# グローバル状態（JS state 完全対応）
# =========================================================
running = True

# モード
is_calibrating = False
tracking = False

# 入力
latest_gaze = None            # {"x","y","confidence"}
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

# ===== Serial command snapshot =====
cmd_pan = None
cmd_tilt = None
cmd_z = None
cmd_lock = threading.Lock()


# ===== Serial =====
serial_port = None      # 選択されたポート名
serial_connected = False

# ===== Z UI =====
Z_MIN = 0
Z_MAX = 1000
manual_z = None         # None = auto


# ===== Depth calibration (observed_z -> true_mm) =====
DEPTH_CALIB_POINTS_MM = [500, 1000, 2000]     # 0.5m / 1.0m / 2.0m
DEPTH_SAMPLE_SEC = 2.5                        # 1点あたりサンプリング秒
DEPTH_MIN_FIX_CONF = 0.0                      # 0でOK（必要なら0.6などに）

depth_calib_observed = {}  # { true_mm: observed_z_median }
depth_calib_enabled = True

# UI表示用
depth_sampling = False
depth_sampling_target = None
depth_sampling_count = 0
depth_sampling_last_obs = None

PAN_TILT_SAMPLE_SEC = 0.7      # サンプリング時間
PAN_TILT_MIN_CONF = 0.4        # gaze confidence 閾値

# ===== iPhone (ZigSim) input =====
iphone_active = False
iphone_last_t = 0.0

iphone_touch_x = 0.0
iphone_touch_y = 0.0
iphone_pitch = None   # radians or degrees

def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def list_serial_ports():
    return [p.device for p in list_ports.comports()]

def save_calibration_json(path="calibration.json"):
    if model is None:
        print("[CALIB SAVE] no pan/tilt model")
        return

    Mp, Mt = model

    data = {
        "pan_tilt_model": {
            "Mp": Mp.tolist(),
            "Mt": Mt.tolist(),
        },
        "depth_calib": depth_calib_observed,
        "meta": {
            "timestamp": time.time()
        }
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[CALIB SAVE] saved to {path}")

def load_calibration_json(path="calibration.json"):
    global model, depth_calib_observed

    if not os.path.exists(path):
        print("[CALIB LOAD] file not found:", path)
        return

    with open(path, "r") as f:
        data = json.load(f)

    # pan / tilt model
    pt = data.get("pan_tilt_model")
    if pt:
        Mp = np.array(pt["Mp"], dtype=float)
        Mt = np.array(pt["Mt"], dtype=float)
        model = (Mp, Mt)
        print("[CALIB LOAD] pan/tilt model loaded")

    # depth calib
    dc = data.get("depth_calib")
    if dc:
        # JSON は key が str になるので int に戻す
        depth_calib_observed = {int(k): float(v) for k, v in dc.items()}
        print("[CALIB LOAD] depth calib loaded:", depth_calib_observed)



def on_touch_x(addr, x):
    global iphone_touch_x, iphone_active, iphone_last_t
    iphone_touch_x = float(x)
    iphone_active = True
    iphone_last_t = time.perf_counter()

def on_touch_y(addr, y):
    global iphone_touch_y, iphone_active, iphone_last_t
    iphone_touch_y = float(y)
    iphone_active = True
    iphone_last_t = time.perf_counter()

def get_iphone_touch():
    # [-1,1] → [0,1]
    x = iphone_touch_x
    y = iphone_touch_y
    # x = (iphone_touch_x + 1.0) * 0.5
    # y = (iphone_touch_y + 1.0) * 0.5
    return -x, y

def on_quat(addr, x, y, z, w):
    global iphone_pitch, iphone_active, iphone_last_t

    # pitch（右手系想定）
    pitch = math.asin(2.0 * (w * x - y * z))
    iphone_pitch = x

    iphone_active = True
    iphone_last_t = time.perf_counter()

def quaternion_to_pitch(qx, qy, qz, qw):
    # pitch (X-axis rotation)
    sinp = 2.0 * (qw * qx - qz * qy)
    sinp = max(-1.0, min(1.0, sinp))
    return math.asin(sinp)


def osc_debug(addr, *args):
    print("[OSC]", addr, args)

def handle_zigsim_osc(addr, args):
    global iphone_touch_x, iphone_touch_y
    global iphone_pitch
    global iphone_mode_until

    now = time.perf_counter()

    if addr == "/ZIGSIM/iphone/touch01":
        iphone_touch_x = float(args[0])
        iphone_mode_until = now + IPHONE_HOLD_SEC

    elif addr == "/ZIGSIM/iphone/touch02":
        iphone_touch_y = float(args[0])
        iphone_mode_until = now + IPHONE_HOLD_SEC
        
    elif addr == "/ZIGSIM/iphone/quaternion":
        qx, qy, qz, qw = args
        iphone_pitch = quaternion_to_pitch(qx, qy, qz, qw)
        iphone_last_t = time.perf_counter()
        

def zigsim_thread():
    disp = dispatcher.Dispatcher()
    disp.set_default_handler(
        lambda addr, *args: handle_zigsim_osc(addr, args)
    ) 
    server = osc_server.ThreadingOSCUDPServer(
        ("0.0.0.0", 9000), disp
    )
    print("[ZIGSIM] listening on 9000")
    server.serve_forever()


def touch_to_pan_tilt(x, y):
    # x,y: 0..1
    pan = 180-map_touch_to_deg(x)
    tilt = map_touch_to_deg(y)
    return pan, tilt

def pitch_to_zcmd(pitch):
    if pitch is None:
        return None

    PITCH_MIN = -0.3
    PITCH_MAX =  0.7

    Z_MIN = 300
    Z_MAX = 2000

    # clamp
    p = max(PITCH_MIN, min(PITCH_MAX, pitch))

    # normalize 0..1
    t = (p - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)

    # map to Z
    return int(Z_MIN + t * (Z_MAX - Z_MIN))


def map_touch_to_deg(v):
    # v: -1.0 .. 1.0
    v = max(-1.0, min(1.0, v))
    return (v + 1.0) * 90.0   # → 0 .. 180

def is_iphone_active():
    return time.perf_counter() < iphone_mode_until

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
    Z_FAR = 2000

    # clamp
    z = max(Z_NEAR, min(Z_FAR, z_mm))

    # 近い → 大、遠い → 小
    ratio = (z - Z_NEAR) / (Z_FAR - Z_NEAR)
    return int(ratio * 1000)


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
    sub.setsockopt_string(zmq.SUBSCRIBE, "fixation")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    while running:
        socks = dict(poller.poll(10))
        if sub in socks:
            topic, payload = sub.recv_multipart()
            msg = msgpack.loads(payload, raw=False)

            if topic.startswith(b"gaze"):
                # print(msg.get("gaze_point_3d"))
                n = msg.get("norm_pos")
                conf = msg.get("confidence", 1.0)
                if n and len(n) == 2:
                    latest_gaze = {
                        "x": float(n[0]),
                        "y": float(n[1]),
                        "confidence": float(conf),
                        "t": time.perf_counter()
                    }

            if topic.startswith(b"fixation"):
                if msg.get("method") != "3d gaze":
                    continue
                gp = msg.get("gaze_point_3d")
                print(gp)
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
    global debug_fix_present, debug_fix_z_mm, debug_fix_conf, debug_fix_age

    last_pan = None
    last_tilt = None
    last_pitch = None

    while running:
        debug_fix_present = False
        debug_fix_z_mm = None
        debug_fix_conf = None
        debug_fix_age = None

        # =========================
        # Pan / Tilt
        # =========================
        
        # ===== Input priority =====
        if is_iphone_active():
            # iPhone manual mode
            tx, ty = get_iphone_touch()
            target_pan = map_touch_to_deg(tx)
            target_tilt = map_touch_to_deg(ty)
            print("[CONTROLLER] iPhone touch mode:", tx, "->", target_pan,",", ty,"->", target_tilt)
            # Z 制御（iPhone）
            # if is_iphone_active() and iphone_pitch is not None:
            #     target_z = pitch_to_zcmd(iphone_pitch)


        elif tracking and model and latest_gaze:
            if latest_gaze["confidence"] >= GAZE_CONF_TH:
                x = latest_gaze["x"]
                y = latest_gaze["y"]

                pan, tilt = predict_affine(model, x, y)

                if last_pan is not None:
                    pan = last_pan + SMOOTHING * (pan - last_pan)
                    tilt = last_tilt + SMOOTHING * (tilt - last_tilt)

                target_pan = pan
                target_tilt = tilt
            else:
                # 瞬き中 → 固定
                if last_pan is not None:
                    target_pan = last_pan
                    target_tilt = last_tilt

        else:
            # ★ ここが欠けていた
            target_pan = manual_pan
            target_tilt = manual_tilt

        # =========================
        # Z
        # =========================
        
        if is_iphone_active() and iphone_pitch is not None:
            if last_pitch is None or abs(iphone_pitch - last_pitch) > Z_PITCH_EPS:
                target_z = pitch_to_zcmd(iphone_pitch)
                last_pitch = iphone_pitch
        elif manual_z is not None:
            target_z = manual_z
        elif latest_fix:
            age = (time.perf_counter() - latest_fix["t"]) * 1000
            if age <= FIXATION_STALE_MS:
                observed_z = latest_fix["z"]  # gp[2]

                # depth calib を適用
                if depth_calib_enabled and len(depth_calib_observed) >= 2:
                    true_mm = interp_observed_to_true_mm(observed_z)
                    target_z = true_mm_to_zcmd(true_mm)
                else:
                    # 従来の荒いマップ（保険）
                    target_z = map_z_mm_to_servo(observed_z)

                
        if latest_fix:
            debug_fix_present = True
            debug_fix_z_mm = latest_fix["z"]
            debug_fix_conf = latest_fix["confidence"]
            debug_fix_age = (time.perf_counter() - latest_fix["t"]) * 1000
            
        # ===== 確定コマンドをスナップショット =====
        with cmd_lock:
            cmd_pan = int(round(target_pan))
            cmd_tilt = int(round(target_tilt))
            cmd_z = target_z
                
        # === safety clamp (最終段) ===
        target_pan = clamp(target_pan, PAN_MIN, PAN_MAX)
        target_tilt = clamp(target_tilt, TILT_MIN, TILT_MAX)

        last_pan = target_pan
        last_tilt = target_tilt
        time.sleep(0.01)

def interp_observed_to_true_mm(observed_z: float) -> float:
    """
    depth_calib_observed に基づき observed_z -> true_mm を区分線形で補正。
    2点以上ない場合は observed_z をそのまま返す（=補正なし）。
    """
    if observed_z is None:
        return None

    if len(depth_calib_observed) < 2:
        return float(observed_z)

    # (observed_z, true_mm) のリストにして observed_z でソート
    pairs = sorted([(oz, tm) for tm, oz in depth_calib_observed.items()], key=lambda x: x[0])

    # 端より外は端の区間で外挿
    if observed_z <= pairs[0][0]:
        (x0, y0), (x1, y1) = pairs[0], pairs[1]
        return y0 + (observed_z - x0) * (y1 - y0) / (x1 - x0 + 1e-9)

    if observed_z >= pairs[-1][0]:
        (x0, y0), (x1, y1) = pairs[-2], pairs[-1]
        return y0 + (observed_z - x0) * (y1 - y0) / (x1 - x0 + 1e-9)

    # 区間を見つけて補間
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        if x0 <= observed_z <= x1:
            return y0 + (observed_z - x0) * (y1 - y0) / (x1 - x0 + 1e-9)

    return float(observed_z)


def true_mm_to_zcmd(true_mm: float) -> int:
    """
    真距離(mm) -> Zcmd(0..1000) に変換。
    ここは作品都合でいじる場所。
    """
    Z_NEAR = 300   # 300mm未満は近すぎ扱い
    Z_FAR  = 2000  # 2m以上は遠すぎ扱い
    z = max(Z_NEAR, min(Z_FAR, float(true_mm)))
    return int((z - Z_NEAR) / (Z_FAR - Z_NEAR) * 1000)


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

        with cmd_lock:
            if cmd_pan is None:
                time.sleep(0.01)
                continue
            pan = cmd_pan
            tilt = cmd_tilt
            z = cmd_z


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
        
        ttk.Label(self, text="Depth Calib").pack(pady=(8,2))

        btns = ttk.Frame(self)
        btns.pack()

        ttk.Button(btns, text="Calib @ 0.5m", command=lambda: self.start_depth_calib(500)).grid(row=0, column=0, padx=3)
        ttk.Button(btns, text="Calib @ 1.0m", command=lambda: self.start_depth_calib(1000)).grid(row=0, column=1, padx=3)
        ttk.Button(btns, text="Calib @ 2.0m", command=lambda: self.start_depth_calib(2000)).grid(row=0, column=2, padx=3)

        self.depth_status = ttk.Label(self, text="depth: (not calibrated)")
        self.depth_status.pack(pady=2)

        self.depth_table = ttk.Label(self, text="")
        self.depth_table.pack(pady=2)

        ttk.Button(self, text="Z Auto (Fixation)", command=self.set_z_auto).pack(fill="x")

        ttk.Button(self, text="Add Point", command=self.add_point).pack(fill="x", pady=4)
        ttk.Button(self, text="Fit Model", command=self.fit).pack(fill="x")

        ttk.Button(self, text="Save Calib (JSON)", command=self.save_calib).pack(fill="x", pady=(6,0))
        ttk.Button(self, text="Load Calib (JSON)", command=self.load_calib).pack(fill="x")
        
        self.lbl = ttk.Label(self, text="status")
        self.lbl.pack(pady=6)
        
        self.gaze_lbl = ttk.Label(self, text="gaze ---")
        self.gaze_lbl.pack(pady=2)
        
        self.fix_lbl = ttk.Label(self, text="fix: ---")
        self.fix_lbl.pack(pady=2)

        self.zcmd_lbl = ttk.Label(self, text="Zcmd: ---")
        self.zcmd_lbl.pack(pady=2)
        
        self.input_lbl = ttk.Label(self, text="input: ---")
        self.input_lbl.pack()
        
        if is_iphone_active():
            src = "iPhone"
        elif tracking:
            src = "Pupil"
        else:
            src = "Manual"
        self.input_lbl.config(text=f"input: {src}")


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
        th = threading.Thread(target=self._add_point_worker, daemon=True)
        th.start()

    def _add_point_worker(self):
        global calib_points

        samples_x = []
        samples_y = []

        t_end = time.perf_counter() + PAN_TILT_SAMPLE_SEC

        while time.perf_counter() < t_end and running:
            if latest_gaze:
                conf = latest_gaze.get("confidence", 1.0)
                if conf >= PAN_TILT_MIN_CONF:
                    samples_x.append(latest_gaze["x"])
                    samples_y.append(latest_gaze["y"])
            time.sleep(0.01)

        if len(samples_x) < 5:
            print("[ADD POINT] not enough samples")
            return

        # median
        samples_x.sort()
        samples_y.sort()
        mx = samples_x[len(samples_x)//2]
        my = samples_y[len(samples_y)//2]

        calib_points.append((mx, my, manual_pan, manual_tilt))

        print(
            f"[ADD POINT] gaze=({mx:.3f},{my:.3f}) "
            f"pan={manual_pan:.1f} tilt={manual_tilt:.1f} "
            f"(n={len(samples_x)})"
        )


    def fit(self):
        global model
        if len(calib_points) >= 3:
            model = fit_affine(calib_points)
            print("MODEL FITTED")

    
    def update_ui(self):
        if latest_gaze:
            self.gaze_lbl.config(
                text=(
                    f"gaze x={latest_gaze['x']:.3f} "
                    f"y={latest_gaze['y']:.3f} "
                    f"conf={latest_gaze['confidence']:.2f}"
                )
            )

        if serial_connected:
            self.serial_status.config(text=f"Serial: CONNECTED ({serial_port})")
            
        # ===== fixation display =====
        if debug_fix_present:
            self.fix_lbl.config(
                text=(
                    f"fix: z={debug_fix_z_mm:.0f}mm "
                    f"conf={debug_fix_conf:.2f} "
                    f"age={debug_fix_age:.0f}ms"
                )
            )
        else:
            self.fix_lbl.config(text="fix: NONE")

        # ===== Z command display =====
        if target_z is not None:
            self.zcmd_lbl.config(text=f"Zcmd: {target_z}")
        else:
            self.zcmd_lbl.config(text="Zcmd: ---")
            
        # depth calib status
        if depth_sampling:
            self.depth_status.config(
                text=f"depth sampling: target={depth_sampling_target}mm  n={depth_sampling_count}  last_obs={depth_sampling_last_obs}"
            )
        else:
            self.depth_status.config(text="depth: idle")

        # show table
        if len(depth_calib_observed) > 0:
            rows = []
            for tm in sorted(depth_calib_observed.keys()):
                rows.append(f"{tm}mm -> obs_z={depth_calib_observed[tm]:.1f}")
            self.depth_table.config(text=" | ".join(rows))
        else:
            self.depth_table.config(text="(no depth calib yet)")

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
        
    def start_depth_calib(self, true_mm: int):
    # 別スレッドでサンプリング（UI固まらない）
        th = threading.Thread(target=self._depth_calib_worker, args=(true_mm,), daemon=True)
        th.start()
        
    def save_calib(self):
        save_calibration_json("calibration.json")

    def load_calib(self):
        load_calibration_json("calibration.json")


    def _depth_calib_worker(self, true_mm: int):
        global depth_sampling, depth_sampling_target, depth_sampling_count, depth_sampling_last_obs
        global depth_calib_observed

        depth_sampling = True
        depth_sampling_target = true_mm
        depth_sampling_count = 0
        depth_sampling_last_obs = None

        samples = []
        t_end = time.perf_counter() + DEPTH_SAMPLE_SEC

        while time.perf_counter() < t_end and running:
            if latest_fix:
                age = (time.perf_counter() - latest_fix["t"]) * 1000
                if age <= FIXATION_STALE_MS:
                    conf = float(latest_fix.get("confidence", 1.0))
                    if conf >= DEPTH_MIN_FIX_CONF:
                        oz = float(latest_fix["z"])
                        samples.append(oz)
                        depth_sampling_last_obs = oz
                        depth_sampling_count = len(samples)
            time.sleep(0.01)

        depth_sampling = False

        if len(samples) < 5:
            print("[DEPTH] not enough samples")
            return

        samples.sort()
        median = samples[len(samples)//2]
        depth_calib_observed[true_mm] = float(median)

        print(f"[DEPTH] saved true={true_mm}mm -> observed_z(median)={median:.1f} (n={len(samples)})")


# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    threading.Thread(target=pupil_thread, daemon=True).start()
    threading.Thread(target=controller_thread, daemon=True).start()
    threading.Thread(target=serial_thread, daemon=True).start()
    threading.Thread(target=zigsim_thread, daemon=True).start()  # ← これ

    App().mainloop()
    running = False
