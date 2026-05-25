import os
import cv2
import json
import time
import queue
import signal
import shutil
import threading
import subprocess
from datetime import datetime
import base64
import requests
import numpy as np
import customtkinter as ct
from PIL import Image

from Exercises import get_exercise


MODEL_NAME = "movenet_singlepose_lightning.tflite"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "models", MODEL_NAME)

REFERENCE_SUMMARY_PATH = os.path.join(
    PROJECT_DIR,
    "reference_data",
    "bicep_curl_demo",
    "reference_summary.json",
)

LIVE_OUTPUT_PATH = os.path.join(
    PROJECT_DIR,
    "reference_data",
    "bicep_curl_demo",
    "live_analysis.json",
)


EDGES = {
    (5, 7): "c",
    (7, 9): "c",
    (6, 8): "y",
    (8, 10): "y",
    (5, 11): "c",
    (6, 12): "y",
    (6, 5): "g",
    (12, 11): "g",
    (12, 14): "y",
    (11, 13): "c",
    (14, 16): "y",
    (13, 15): "c",
}

COLOR_MAP = {
    "m": (255, 0, 255),
    "c": (255, 255, 0),
    "y": (0, 255, 255),
    "g": (26, 25, 25),
}


# ================= RPiCam Stream =================

class RPiCamMJPEGStream:
    def __init__(
        self,
        width=640,
        height=480,
        framerate=15,
        timeout=0,
        camera_index=0,
    ):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.timeout = timeout
        self.camera_index = camera_index
        self.process = None
        self.buffer = b""

    def open(self):
        if shutil.which("rpicam-vid") is None:
            raise RuntimeError(
                "rpicam-vid not found. Install/enable Raspberry Pi camera tools first."
            )

        command = [
            "rpicam-vid",
            "--camera", str(self.camera_index),
            "--timeout", str(self.timeout),
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.framerate),
            "--codec", "mjpeg",
            "--nopreview",
            "--output", "-",
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            preexec_fn=os.setsid,
        )

        time.sleep(1.0)

        if self.process.poll() is not None:
            stderr = self.process.stderr.read().decode(errors="ignore")
            raise RuntimeError(f"rpicam-vid failed to start:\n{stderr}")

        return True

    def is_opened(self):
        return self.process is not None and self.process.poll() is None

    def read(self):
        if not self.is_opened():
            return False, None

        while True:
            chunk = self.process.stdout.read(4096)

            if not chunk:
                return False, None

            self.buffer += chunk

            start = self.buffer.find(b"\xff\xd8")
            end = self.buffer.find(b"\xff\xd9")

            if start != -1 and end != -1 and end > start:
                jpg = self.buffer[start:end + 2]
                self.buffer = self.buffer[end + 2:]

                img_array = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is not None:
                    return True, frame

    def release(self):
        if self.process is None:
            return

        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=2)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass

        self.process = None


# ================= GUI IMAGE =================

class CTkImageDisplay(ct.CTkLabel):
    DISPLAY_WIDTH = 550
    DISPLAY_HEIGHT = 360

    def __init__(self, master):
        super().__init__(master, text="")

    def set_frame(self, frame):
        frame = cv2.resize(frame, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        ctk_img = ct.CTkImage(
            light_image=img,
            dark_image=img,
            size=(self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT),
        )

        self.configure(image=ctk_img)
        self.image = ctk_img


class MyTabView(ct.CTkTabview):
    def __init__(self, master, feedback_text, programme_text, **kwargs):
        super().__init__(master, width=230, height=360, **kwargs)

        self.add("Programme")
        self.add("Live Feedback")

        text_panel_width = 210
        text_panel_height = 305

        self.programme_label = ct.CTkLabel(
            master=self.tab("Programme"),
            fg_color="#03045E",
            text=programme_text,
            font=("Roboto", 15),
            text_color="white",
            wraplength=190,
            justify="left",
            anchor="nw",
        )
        self.programme_label.configure(
            width=text_panel_width,
            height=text_panel_height,
        )
        self.programme_label.pack(fill="both", expand=True, padx=6, pady=6)

        self.feedback_label = ct.CTkLabel(
            master=self.tab("Live Feedback"),
            fg_color="#03045E",
            text=feedback_text,
            font=("Roboto", 16),
            text_color="white",
            wraplength=190,
            justify="left",
            anchor="nw",
        )
        self.feedback_label.configure(
            width=text_panel_width,
            height=text_panel_height,
        )
        self.feedback_label.pack(fill="both", expand=True, padx=6, pady=6)


# ================= MAIN TRACKER =================

class ExerciseTracker:
    def __init__(
        self,
        master,
        home_callback=None,
        exercise_id="bicep_curl_demo",
        side="right",
    ):
        self.master = master
        self.home_callback = home_callback

        self.enabled = True
        self.feedback_text = "Waiting for movement..."
        self.score = 0

        self.active_set = 1
        self.active_rep = 0
        self.target_sets = 2
        self.target_reps = 5

        self.exercise = get_exercise(
            exercise_id=exercise_id,
            reference_summary_path=REFERENCE_SUMMARY_PATH,
            side=side,
        )

        self.input_size = 192
        self.queue = queue.Queue(maxsize=1)
        self.camera = None

        self.fps = 0.0
        self.inference_latency_ms = 0.0
        self.last_fps_time = time.time()
        self.processed_frames = 0

        self.draw_widgets()

    def draw_widgets(self):
        self.frame = ct.CTkFrame(self.master, fg_color="#f0f4f7")

        # ================= HEADER =================
        header_frame = ct.CTkFrame(self.frame, fg_color="#99ddff", height=86)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)

        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.home_icon = ct.CTkImage(
            Image.open(os.path.join(script_dir, "home.png")),
            size=(50, 50),
        )

        self.home_button = ct.CTkButton(
            header_frame,
            image=self.home_icon,
            corner_radius=None,
            text="",
            command=self.on_home_clicked,
            fg_color="#99ddff",
            hover_color="#99ddff",
            width=70,
            height=70,
        )
        self.home_button.pack(side="left", padx=6, pady=8)

        self.title_label = ct.CTkLabel(
            header_frame,
            text="Exercise Tracker",
            font=("Roboto", 32),
            text_color="dark blue",
        )
        self.title_label.place(relx=0.5, rely=0.5, anchor=ct.CENTER)

        self.settings_icon = ct.CTkImage(
            Image.open(os.path.join(script_dir, "gear.png")),
            size=(50, 50),
        )

        self.button_image = ct.CTkButton(
            header_frame,
            image=self.settings_icon,
            corner_radius=None,
            text="",
            command=lambda: print("Settings clicked"),
            fg_color="#99ddff",
            hover_color="#99ddff",
            width=70,
            height=70,
        )
        self.button_image.pack(side=ct.RIGHT, padx=6, pady=8)

        self.date_time = datetime.now().strftime("%A %H:%M\n%d/%m/%Y")
        self.label_date = ct.CTkLabel(
            header_frame,
            text=self.date_time,
            font=("Roboto", 15),
            text_color="dark blue",
        )
        self.label_date.pack(side=ct.RIGHT, padx=5)

        # ================= BODY =================
        self.body_frame = ct.CTkFrame(self.frame, fg_color="#ffffff")
        self.body_frame.pack(fill="both", expand=True)

        self.image_display = CTkImageDisplay(self.body_frame)
        self.image_display.pack(side=ct.LEFT, padx=5, pady=5)

        self.info_frame = ct.CTkFrame(
            self.body_frame,
            fg_color="transparent",
            width=235,
        )
        self.info_frame.pack(side=ct.RIGHT, fill="both", expand=True, padx=5, pady=5)
        self.info_frame.pack_propagate(False)

        programme_text = (
            f"{self.exercise.exercise_name}\n\n"
            "Goal:\n"
            "- Keep elbow close to your side\n"
            "- Curl the forearm upward slowly\n"
            "- Lower the arm with control\n"
            "- Avoid swinging the shoulder"
        )

        self.tab_view = MyTabView(
            master=self.info_frame,
            feedback_text=self.feedback_text,
            programme_text=programme_text,
            fg_color="#ffffff",
        )
        self.tab_view.pack(fill="both", expand=True)

        self.webcam_thread = threading.Thread(
            target=self.run_live,
            daemon=True,
        )
        self.webcam_thread.start()

        self.master.after(20, self.update_gui)

    def run_inference(self, frame):
        start = time.perf_counter()

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        response = requests.post(
            "http://127.0.0.1:5055/infer",
            json={"image": image_b64},
            timeout=2,
        )

        response.raise_for_status()

        self.inference_latency_ms = (time.perf_counter() - start) * 1000

        keypoints = response.json()["keypoints"]
        return np.array(keypoints, dtype=np.float32)

    def draw_skeleton(self, frame, joints):
        def to_xy(joint):
            if isinstance(joint, dict):
                return int(joint["x"]), int(joint["y"])
            return int(joint[0]), int(joint[1])

        for (p1, p2), colour_key in EDGES.items():
            if p1 in joints and p2 in joints:
                pt1 = to_xy(joints[p1])
                pt2 = to_xy(joints[p2])
                cv2.line(frame, pt1, pt2, COLOR_MAP[colour_key], 2)

        for joint in joints.values():
            point = to_xy(joint)
            cv2.circle(frame, point, 4, (255, 0, 255), -1)

    def overlay_status(self, frame):
        # Slightly larger panel to fit bigger text
        cv2.rectangle(frame, (10, 10), (230, 150), (0, 0, 0), -1)

        x = 20

        cv2.putText(frame, f"{self.exercise.exercise_name}",
                    (x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2)

        cv2.putText(frame, f"Score: {int(self.score)}",
                    (x, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (0, 255, 0), 2)

        cv2.putText(frame, f"Set: {self.active_set}/{self.target_sets}",
                    (x, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 255, 255), 2)

        cv2.putText(frame, f"Rep: {self.active_rep}/{self.target_reps}",
                    (x, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 255, 255), 2)

        cv2.putText(frame, f"{self.fps:.1f} FPS | {self.inference_latency_ms:.0f} ms",
                    (x, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (255, 255, 255), 1)

    def run_live(self):
        frame_count = 0

        # Performance tracking
        self.fps = 0.0
        self.inference_latency_ms = 0.0
        self.processed_frames = 0
        self.last_fps_time = time.time()

        try:
            self.camera = RPiCamMJPEGStream(
                width=640,
                height=480,
                framerate=15,
                timeout=0,
                camera_index=0,
            )
            self.camera.open()

        except Exception as e:
            self.feedback_text = f"Camera error: {e}"
            print(self.feedback_text)
            return

        while self.enabled and self.camera.is_opened():
            ret, frame = self.camera.read()

            if not ret or frame is None:
                self.feedback_text = "Camera frame not received."
                break

            frame_count += 1

            # Process every second frame to reduce Raspberry Pi workload.
            if frame_count % 2 != 0:
                continue

            try:
                # Measure local Flask/MoveNet inference latency.
                inference_start = time.perf_counter()
                keypoints = self.run_inference(frame)
                self.inference_latency_ms = (time.perf_counter() - inference_start) * 1000

                # Estimate processed FPS.
                self.processed_frames += 1
                now = time.time()

                if now - self.last_fps_time >= 1.0:
                    self.fps = self.processed_frames / (now - self.last_fps_time)
                    self.processed_frames = 0
                    self.last_fps_time = now

                joints = self.exercise.extract_joints(
                    keypoints=keypoints,
                    frame_shape=frame.shape,
                )

                # Run exercise analysis every third frame.
                if frame_count % 3 == 0:
                    analysis = self.exercise.analyse(joints)

                    self.score = self.exercise.compute_score(analysis)
                    self.feedback_text = self.exercise.feedback(
                        analysis,
                        self.score,
                    )

                    self.active_set = analysis.get(
                        "set_count",
                        getattr(self.exercise, "active_set", self.active_set),
                    )
                    self.active_rep = analysis.get(
                        "rep_count",
                        getattr(self.exercise, "active_rep", self.active_rep),
                    )

                    self.target_sets = getattr(
                        self.exercise,
                        "target_sets",
                        self.target_sets,
                    )
                    self.target_reps = getattr(
                        self.exercise,
                        "target_reps",
                        self.target_reps,
                    )

                self.draw_skeleton(frame, joints)

            except Exception as e:
                self.feedback_text = f"Tracking error: {e}"

            self.overlay_status(frame)

            if not self.queue.full():
                self.queue.put(frame)

        if self.camera:
            self.camera.release()

        self.save_live_results()

    def save_live_results(self):
        os.makedirs(os.path.dirname(LIVE_OUTPUT_PATH), exist_ok=True)

        with open(LIVE_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(self.exercise.score_history, f, indent=2)

    def update_gui(self):
        try:
            frame = self.queue.get_nowait()
            self.image_display.set_frame(frame)
            self.tab_view.feedback_label.configure(text=self.feedback_text)
        except queue.Empty:
            pass

        self.master.after(30, self.update_gui)

    def on_home_clicked(self):
        self.enabled = False

        if self.camera:
            self.camera.release()

        if callable(self.home_callback):
            self.home_callback()

    def show(self):
        self.frame.pack(fill="both", expand=True)

    def hide(self):
        self.enabled = False

        if self.camera:
            self.camera.release()

        self.frame.pack_forget()