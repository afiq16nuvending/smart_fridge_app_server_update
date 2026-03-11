from fastapi import FastAPI, WebSocket
import uvicorn
import cv2
import numpy as np
import argparse
import os
import glob
import json
import time
from datetime import datetime
from pathlib import Path
import hashlib
import io
import tempfile
import subprocess
import threading
import asyncio
from threading import Thread, Lock
from multiprocessing import Event
import multiprocessing
import signal
from collections import deque, defaultdict
from typing import List, Dict, Tuple
import random
import RPi.GPIO as GPIO
import atexit
import requests
from requests.auth import HTTPBasicAuth
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import hailo
from hailo_pipeline import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
    get_default_parser,
    detect_hailo_arch,
    QUEUE,
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE
)
from gtts import gTTS
import pygame
import gc
import psutil
import sys
import ctypes
from scipy.spatial import distance
import weakref
import setproctitle

app = FastAPI()
data_deque: Dict[int, deque] = {}

DOOR_LOCK_PIN = 25
DOOR_SWITCH_PIN = 26
LED_GREEN = 23
LED_RED = 18
BUZZER_PIN = 20

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(LED_GREEN, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(LED_RED, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(DOOR_LOCK_PIN, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(DOOR_SWITCH_PIN, GPIO.IN)

readyToProcess = False
blink = False
alert_thread = None
camera_covered = False
cover_alert_thread = None
camera_covered_sound_playing = False
price_alert_sound_playing = False
last_alerted_label = None
current_pipeline_app = None
pipeline_lock = threading.Lock()
print_lock = threading.Lock()
done = False
unlock_data = 0

movement_history = defaultdict(lambda: deque(maxlen=5))
bbox_area_history = defaultdict(lambda: deque(maxlen=10))
movement_direction = {}
last_counted_direction = {}
object_trails = defaultdict(lambda: deque(maxlen=30))
global_trails = defaultdict(lambda: deque(maxlen=30))

global_track_counter = 0
local_to_global_id_map = {}
global_movement_history = defaultdict(deque)
global_last_counted_direction = {}
global_track_labels = {}
active_objects_per_camera = {0: defaultdict(dict), 1: defaultdict(dict)}
cross_camera_candidates = defaultdict(list)
camera_movement_history = {
    0: defaultdict(lambda: deque(maxlen=5)),
    1: defaultdict(lambda: deque(maxlen=5))
}
camera_bbox_area_history = {
    0: defaultdict(lambda: deque(maxlen=5)),
    1: defaultdict(lambda: deque(maxlen=5))
}


def trigger_buzzer(duration=0.5):
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)


def blink_led(pin, times, delay):
    for _ in range(times):
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(delay)


def control_door(pin, action, duration=0.5):
    if action.lower() == 'unlock':
        GPIO.output(pin, GPIO.LOW)
        time.sleep(duration)
        GPIO.output(pin, GPIO.HIGH)
    elif action.lower() == 'lock':
        GPIO.output(pin, GPIO.HIGH)


def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    if label == 0:
        color = (85, 45, 255)
    elif label == 2:
        color = (222, 82, 175)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_trail(frame, track_id, center, color, global_id=None):
    if global_id is not None:
        global_trails[global_id].appendleft(center)
        points = list(global_trails[global_id])
    else:
        object_trails[track_id].appendleft(center)
        points = list(object_trails[track_id])
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        thickness = int(np.sqrt(64 / float(i + 1)) * 2)
        cv2.line(frame, points[i - 1], points[i], color, thickness)


def draw_counts(frame, class_counters, label):
    class_names = {
        0: "", 1: "chickenKatsuCurry", 2: "dakgangjeongRice",
        3: "dragonFruit", 4: "guava", 5: "kimchiFriedRice",
        6: "kimchiTuna", 7: "mango", 8: "mangoMilk",
        9: "pineappleHoney", 10: "pinkGuava",
    }
    total_entry = sum(class_counters["entry"].values())
    total_exit = sum(class_counters["exit"].values())
    cv2.putText(frame, f'Total Entry: {total_entry}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Total Exit: {total_exit}', (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y_offset = 110
    all_labels = set(class_counters["entry"].keys()) | set(class_counters["exit"].keys())
    for label in all_labels:
        entry_count = class_counters["entry"].get(label, 0)
        exit_count = class_counters["exit"].get(label, 0)
        class_id = next(k for k, v in class_names.items() if v == label)
        color = compute_color_for_labels(class_id)
        text = f'{label} Entry: {entry_count}, Exit: {exit_count}'
        cv2.putText(frame, text, (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 30


def draw_zone(frame):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
    cv2.putText(frame, "Detection Zone", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def handle_alert_state():
    global blink
    while blink:
        if GPIO.input(DOOR_SWITCH_PIN) == 0:
            GPIO.output(LED_RED, GPIO.LOW)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            break
        GPIO.output(LED_RED, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(LED_RED, GPIO.LOW)
        time.sleep(0.5)


def calculate_total_price_and_control_buzzer(current_data, deposit, label=None):
    global blink, alert_thread, price_alert_sound_playing, last_alerted_label
    total_product_price = 0
    validated_products = current_data.get("validated_products", {})
    all_products = set(validated_products.get("entry", {}).keys()) | \
                   set(validated_products.get("exit", {}).keys())
    product_prices = {}
    for product_name in all_products:
        entry_data = validated_products.get("entry", {}).get(product_name, {"count": 0})
        exit_data = validated_products.get("exit", {}).get(product_name, {"count": 0})
        entry_count = entry_data.get("count", 0)
        exit_count = exit_data.get("count", 0)
        product_details = exit_data.get("product_details") or entry_data.get("product_details")
        if product_details and "product_price" in product_details:
            price_per_unit = float(product_details["product_price"])
            true_count = max(0, exit_count - entry_count)
            product_total = true_count * price_per_unit
            if true_count > 0:
                product_prices[product_name] = product_total
                total_product_price += product_total
    return total_product_price


def is_frame_dark(frame, threshold=40):
    global camera_covered_sound_playing
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    avg_brightness = np.mean(gray)
    return avg_brightness < threshold


def setup_cover_alert_sound():
    alert_dir = "sounds/cover_alerts"
    alert_file = os.path.join(alert_dir, "camera_covered.mp3")
    os.makedirs(alert_dir, exist_ok=True)
    if not os.path.exists(alert_file):
        alert_text = "Dont cover the camera. Please uncover the camera immediately."
        tts = gTTS(text=alert_text, lang='en', slow=False)
        tts.save(alert_file)
    return alert_file


def handle_cover_alert():
    global camera_covered
    alert_sound = setup_cover_alert_sound()
    while camera_covered:
        if GPIO.input(DOOR_SWITCH_PIN) == 0:
            break
        tts_manager.play_mp3_async(alert_sound, volume=0.8)
        time.sleep(3.0)


def check_door_status():
    while True:
        door_sw = GPIO.input(DOOR_SWITCH_PIN)
        with print_lock:
            if door_sw == 0:
                return True
        time.sleep(0.1)


class DualCameraRecorder:
    def __init__(self, transaction_id, cam0_device=0, cam1_device=2):
        self.transaction_id = transaction_id
        self.cam0_device = cam0_device
        self.cam1_device = cam1_device
        self.is_recording = False
        self.shutdown_event = Event()
        self.cam0_video_path = None
        self.cam1_video_path = None
        self.cam0_frame_count = 0
        self.cam1_frame_count = 0
        self.cam0_thread = None
        self.cam1_thread = None
        self.video_dir = os.path.join(os.getcwd(), "saved_videos")
        os.makedirs(self.video_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.cam0_video_path = os.path.join(self.video_dir, f"cam0_{timestamp}_{transaction_id}.avi")
        self.cam1_video_path = os.path.join(self.video_dir, f"cam1_{timestamp}_{transaction_id}.avi")

    def _record_camera(self, device_id, output_path, camera_name):
        global camera_covered, cover_alert_thread
        cap = None
        writer = None
        frame_count = 0
        fps_start_time = None
        fps_calculated = False
        actual_fps = 15.0
        try:
            cap = cv2.VideoCapture(device_id)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_FPS, 25)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print(f"ERROR: Could not open {camera_name}")
                return
            print(f"[Recording] {camera_name} started -> {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            while not self.shutdown_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                if fps_start_time is None:
                    fps_start_time = time.time()
                frame_count += 1
                if not fps_calculated and frame_count >= 30:
                    actual_fps = frame_count / (time.time() - fps_start_time)
                    fps_calculated = True
                    print(f"[Recording] {camera_name} FPS: {actual_fps:.2f}")
                if writer is None and fps_calculated:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(output_path, fourcc, actual_fps, (w, h), True)
                if writer is not None:
                    writer.write(frame)
                if is_frame_dark(frame):
                    if not camera_covered:
                        camera_covered = True
                        if cover_alert_thread is None or not cover_alert_thread.is_alive():
                            cover_alert_thread = threading.Thread(target=handle_cover_alert, daemon=True)
                            cover_alert_thread.start()
                else:
                    if camera_covered:
                        camera_covered = False
            if camera_name == "Camera 0":
                self.cam0_frame_count = frame_count
            else:
                self.cam1_frame_count = frame_count
        except Exception as e:
            print(f"[Recording] {camera_name} error: {e}")
        finally:
            if writer:
                writer.release()
            if cap:
                cap.release()
            print(f"[Recording] {camera_name}: {frame_count} frames saved")

    def start_recording(self):
        self.is_recording = True
        self.shutdown_event.clear()
        self.cam0_thread = threading.Thread(target=self._record_camera, args=(self.cam0_device, self.cam0_video_path, "Camera 0"), daemon=True)
        self.cam1_thread = threading.Thread(target=self._record_camera, args=(self.cam1_device, self.cam1_video_path, "Camera 1"), daemon=True)
        self.cam0_thread.start()
        self.cam1_thread.start()
        print(f"[Recording] Dual camera recording started")

    def stop_recording(self):
        self.shutdown_event.set()
        self.is_recording = False
        if self.cam0_thread and self.cam0_thread.is_alive():
            self.cam0_thread.join(timeout=5)
        if self.cam1_thread and self.cam1_thread.is_alive():
            self.cam1_thread.join(timeout=5)
        print(f"[Recording] Stopped. Cam0: {self.cam0_frame_count}f, Cam1: {self.cam1_frame_count}f")
        return self.cam0_video_path, self.cam1_video_path


def display_user_data_frame(user_data):
    video_dir = os.path.join(os.getcwd(), "saved_videos")
    os.makedirs(video_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = None
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    transaction_id = getattr(user_data, 'transaction_id', 'unknown')
    filename = os.path.join(video_dir, f"annotated_{timestamp}_{transaction_id}.avi")
    frame_count = 0
    fps_start_time = None
    fps_calculated = False
    actual_fps = 13.0
    try:
        while not user_data.shutdown_event.is_set():
            frame = user_data.get_frame()
            if frame is not None:
                if fps_start_time is None:
                    fps_start_time = time.time()
                frame_count += 1
                if not fps_calculated and frame_count >= 30:
                    actual_fps = frame_count / (time.time() - fps_start_time)
                    fps_calculated = True
                if output_video is None and fps_calculated:
                    height, width = frame.shape[:2]
                    output_video = cv2.VideoWriter(filename, fourcc, actual_fps, (width, height), True)
                if output_video is not None:
                    output_video.write(frame.copy())
                cv2.imshow("Post-Processing Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)
    except Exception as e:
        print(f"Error in display loop: {e}")
    finally:
        if output_video is not None:
            output_video.release()
            print(f"Annotated video saved: {filename} ({frame_count} frames)")
        cv2.destroyAllWindows()
        for i in range(5):
            cv2.waitKey(1)


def stream_video_to_api(video_path, dataset_name, transaction_id, machine_id, user_id, machine_identifier):
    api_url = "https://stg-sfapi.nuboxtech.com/index.php/shopping_app/machine/TransactionDataset/insert_transactionDataset"
    username = 'admin'
    password = '1234'
    api_key = '123456'
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    filename = os.path.basename(video_path)
    payload = {
        'machine_id': machine_id, 'created_by': user_id,
        'dataset_url': f"assets/video/machine_transaction_dataset/{machine_identifier}/{dataset_name}.avi",
        'dataset_name': dataset_name, 'transaction_id': transaction_id,
        'created_datetime': current_time
    }
    headers = {'x-api-key': api_key}
    try:
        with open(video_path, 'rb') as video_file:
            files = {'video': (filename, video_file, 'video/avi')}
            response = requests.post(api_url, auth=HTTPBasicAuth(username, password),
                                     headers=headers, data=payload, files=files, timeout=30.0)
            return response.status_code == 200
    except Exception as e:
        print(f"Error during video streaming: {e}")
        return False


def monitor_and_send_videos(video_directory, machine_id, machine_identifier, user_id):
    processed_videos = set()
    processing_lock = Lock()

    def create_lock_file(video_path):
        lock_path = video_path + '.processing'
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{time.time()}\n".encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False
        except:
            return False

    def remove_lock_file(video_path):
        lock_path = video_path + '.processing'
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except:
            pass

    def is_lock_stale(video_path, timeout=300):
        lock_path = video_path + '.processing'
        try:
            if not os.path.exists(lock_path):
                return False
            with open(lock_path, 'r') as f:
                timestamp = float(f.read().strip())
            return time.time() - timestamp > timeout
        except:
            return True

    while True:
        try:
            video_files = glob.glob(os.path.join(video_directory, "*.avi"))
            for video_path in video_files:
                with processing_lock:
                    if video_path in processed_videos:
                        continue
                if is_lock_stale(video_path):
                    remove_lock_file(video_path)
                if not create_lock_file(video_path):
                    continue
                try:
                    if not os.path.exists(video_path):
                        with processing_lock:
                            processed_videos.add(video_path)
                        continue
                    if is_file_complete_enhanced(video_path):
                        dataset_name = os.path.splitext(os.path.basename(video_path))[0]
                        tid = dataset_name.split('_')[-1] if '_' in dataset_name else 'unknown'
                        if stream_video_to_api(video_path, dataset_name, tid, machine_id, user_id, machine_identifier):
                            try:
                                os.remove(video_path)
                            except:
                                pass
                finally:
                    remove_lock_file(video_path)
                    with processing_lock:
                        processed_videos.add(video_path)
                    if len(processed_videos) > 50:
                        processed_videos = set(list(processed_videos)[-50:])
        except Exception as e:
            print(f"Error in video monitoring: {e}")
        time.sleep(10)


def is_file_complete_enhanced(file_path, stable_time=5):
    try:
        if not os.path.exists(file_path):
            return False
        initial_stat = os.stat(file_path)
        if initial_stat.st_size == 0:
            return False
        time.sleep(stable_time)
        try:
            final_stat = os.stat(file_path)
        except OSError:
            return False
        if initial_stat.st_size != final_stat.st_size or initial_stat.st_mtime != final_stat.st_mtime:
            return False
        try:
            with open(file_path, 'r+b') as f:
                f.seek(0, 2)
                if f.tell() != final_stat.st_size:
                    return False
        except (IOError, OSError):
            return False
        return True
    except:
        return False


class WebSocketDataManager:
    def __init__(self):
        self.current_data = {
            "validated_products": {"entry": {}, "exit": {}},
            "invalidated_products": {"entry": {}, "exit": {}}
        }
        self._lock = threading.Lock()

    def update_data(self, new_data):
        with self._lock:
            self.current_data = new_data

    def get_current_data(self):
        with self._lock:
            return self.current_data.copy()


class TrackingData:
    def __init__(self):
        self.shutdown_event = Event()
        self.validated_products = {"entry": {}, "exit": {}}
        self.invalidated_products = {"entry": {}, "exit": {}}
        self.class_counters = {"entry": defaultdict(int), "exit": defaultdict(int)}
        self.counted_tracks = {"entry": set(), "exit": set()}
        self.machine_planogram = []
        self.hailo_pipeline_string = ""
        self.frame_rate_calc = 1
        self.last_time = time.time()
        self.websocket_data_manager = WebSocketDataManager()
        self.deposit = 0.0
        self.machine_id = None
        self.machine_identifier = None
        self.user_id = None
        self.transaction_id = None

    def set_transaction_data(self, deposit, machine_id, machine_identifier, user_id, transaction_id):
        self.deposit = deposit
        self.machine_id = machine_id
        self.machine_identifier = machine_identifier
        self.user_id = user_id
        self.transaction_id = transaction_id

class HailoDetectionCallback(app_callback_class):
    def __init__(self, websocket=None, deposit=0.0, machine_id=None,
                 machine_identifier=None, user_id=None, transaction_id=None):
        super().__init__()
        self.tracking_data = TrackingData()
        self.use_frame = True
        self.websocket = websocket
        self.shutdown_event = Event()
        self.deposit = deposit
        self.machine_id = machine_id
        self.machine_identifier = machine_identifier
        self.user_id = user_id
        self.transaction_id = transaction_id
        self.tracking_data.set_transaction_data(deposit, machine_id, machine_identifier, user_id, transaction_id)
        self.video_directory = os.path.join(os.getcwd(), "saved_videos")
        os.makedirs(self.video_directory, exist_ok=True)
        self.store_machine_id_env(machine_id)
        self.load_machine_planogram()

    def store_machine_id_env(self, machine_id):
        if machine_id is not None:
            os.environ['MACHINE_ID'] = str(machine_id)

    def load_machine_id_env(self):
        return os.environ.get('MACHINE_ID')

    def is_planogram_valid_for_machine(self, machine_id):
        try:
            stored = os.environ.get('PLANOGRAM_MACHINE_ID')
            return stored == str(machine_id) if stored else False
        except:
            return False

    def store_planogram_env(self, planogram_data):
        try:
            os.environ['MACHINE_PLANOGRAM'] = json.dumps(planogram_data)
            current_id = self.load_machine_id_env()
            if current_id:
                os.environ['PLANOGRAM_MACHINE_ID'] = str(current_id)
            self.tracking_data.machine_planogram = planogram_data
        except Exception as e:
            print(f"Error storing planogram: {e}")

    def load_planogram_env(self):
        try:
            data = os.environ.get('MACHINE_PLANOGRAM')
            return json.loads(data) if data else []
        except:
            return []

    def load_machine_planogram(self):
        try:
            current_id = self.load_machine_id_env()
            if not current_id:
                existing = self.load_planogram_env()
                self.tracking_data.machine_planogram = existing if existing else []
                return
            existing = self.load_planogram_env()
            if existing and self.is_planogram_valid_for_machine(current_id):
                self.tracking_data.machine_planogram = existing
                self.start_planogram_refresh_thread()
                video_thread = threading.Thread(target=monitor_and_send_videos,
                    args=(self.video_directory, current_id, self.machine_identifier, self.user_id), daemon=True)
                video_thread.start()
                return
            self.fetch_and_store_initial_planogram(current_id)
        except Exception as e:
            print(f"Error loading planogram: {e}")
            self.tracking_data.machine_planogram = self.load_planogram_env() or []

    def fetch_and_store_initial_planogram(self, machine_id):
        try:
            username = 'admin'
            password = '1234'
            api_key = '123456'
            headers = {'x-api-key': api_key}
            
            api_endpoint = f'https://stg-sfapi.nuboxtech.com/index.php/'
                           f'mobile_app/machine/Machine_listing/machine_planogram/{machine_id}')
                          
                          
            video_thread = threading.Thread(target=monitor_and_send_videos,
                args=(self.video_directory, machine_id, self.machine_identifier, self.user_id), daemon=True)
            video_thread.start()
            self.start_planogram_refresh_thread()
            response = requests.get(api_endpoint, auth=HTTPBasicAuth(username, password), headers=headers)
            if response.status_code == 200:
                planogram = response.json().get('machine_planogram', [])
                self.store_planogram_env(planogram)
            else:
                self.tracking_data.machine_planogram = []
        except Exception as e:
            print(f"Error fetching planogram: {e}")
            self.tracking_data.machine_planogram = []

    def start_planogram_refresh_thread(self):
        def refresh():
            username = 'admin'
            password = '1234'
            api_key = '123456'
            headers = {'x-api-key': api_key}
            while True:
                try:
                    mid = self.load_machine_id_env()
                    if not mid:
                        time.sleep(1000)
                        continue
                        
                    endpoint = (f'https://stg-sfapi.nuboxtech.com/index.php/'
                                f'mobile_app/machine/Machine_listing/'
                                f'machine_planogram/{refresh_machine_id}')
                    
                    resp = requests.get(endpoint, auth=HTTPBasicAuth(username, password), headers=headers)
                    if resp.status_code == 200:
                        new_data = resp.json().get('machine_planogram', [])
                        if new_data != self.load_planogram_env():
                            self.store_planogram_env(new_data)
                except:
                    pass
                time.sleep(1000)
        threading.Thread(target=refresh, daemon=True).start()

    def get_fallback_pipeline_string(self):
        cam0_video = os.environ.get('POSTPROCESS_CAM0_VIDEO', '')
        cam1_video = os.environ.get('POSTPROCESS_CAM1_VIDEO', '')
        return (
            "hailoroundrobin mode=0 name=fun ! "
            "queue name=hailo_pre_infer_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "hailonet hef-path=resources/ai_model.hef batch-size=2 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! "
            "queue name=hailo_postprocess0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "hailofilter function-name=filter_letterbox so-path=/home/afiq/hailo-rpi5-examples/basic_pipelines/../resources/libyolo_hailortpp_postprocess.so config-path=resources/labels.json qos=false ! "
            "queue name=hailo_track0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "hailotracker name=hailo_tracker class-id=-1 kalman-dist-thr=0.8 iou-thr=0.9 init-iou-thr=0.7 keep-new-frames=1 keep-tracked-frames=1 keep-lost-frames=1 keep-past-metadata=true ! "
            "hailostreamrouter name=sid src_0::input-streams=\"<sink_0>\" src_1::input-streams=\"<sink_1>\" "
            "compositor name=comp start-time-selection=0 sink_0::xpos=0 sink_0::ypos=0 sink_1::xpos=350 sink_1::ypos=0 ! "
            "queue name=hailo_video_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "videoconvert ! "
            "queue name=hailo_display_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "fpsdisplaysink video-sink=ximagesink name=hailo_display sync=false text-overlay=true "
            f"filesrc location={cam0_video} ! "
            "decodebin ! "
            "queue name=source_scale_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "videoscale name=source_videoscale_0 n-threads=2 ! "
            "queue name=source_convert_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "videoconvert n-threads=3 name=source_convert_0 qos=false ! "
            "video/x-raw, format=RGB, pixel-aspect-ratio=1/1 ! "
            "queue name=inference_wrapper_input_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "fun.sink_0 "
            "sid.src_0 ! "
            "queue name=identity_callback_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "identity name=identity_callback_0 ! "
            "queue name=hailo_draw_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "hailooverlay ! "
            "videoscale n-threads=8 ! "
            "video/x-raw,width=640,height=360 ! "
            "queue name=comp_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "comp.sink_0 "
            f"filesrc location={cam1_video} ! "
            "decodebin ! "
            "queue name=source_scale_q_2 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "videoscale name=source_videoscale_2 n-threads=2 ! "
            "queue name=source_convert_q_2 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "videoconvert n-threads=3 name=source_convert_2 qos=false ! "
            "video/x-raw, format=RGB, pixel-aspect-ratio=1/1 ! "
            "queue name=inference_wrapper_input_q_2 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "fun.sink_1 "
            "sid.src_1 ! "
            "queue name=identity_callback_q_1 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "identity name=identity_callback_1 ! "
            "queue name=hailo_draw_1 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "hailooverlay ! "
            "videoscale n-threads=8 ! "
            "video/x-raw,width=640,height=360 ! "
            "queue name=comp_q_1 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "comp.sink_1"
        )

    def validate_detected_product(self, detected_product):
        current_planogram = self.load_planogram_env()
        if current_planogram:
            self.tracking_data.machine_planogram = current_planogram
        normalized = detected_product.replace(' ', '').lower()
        matching = [p for p in self.tracking_data.machine_planogram
                    if p.get('product_name', '').replace(' ', '').lower() == normalized]
        if matching:
            return {"valid": True, "product_details": matching[0],
                    "message": f"{detected_product} validated"}
        else:
            return {"valid": False, "product_details": None,
                    "message": f"{detected_product} not in planogram"}


class HailoDetectionApp:
    def __init__(self, app_callback, user_data):
        self.app_callback = app_callback
        self.user_data = user_data
        self.shutdown_called = False
        self.shutdown_lock = threading.Lock()
        self.use_frame = True
        self.labels_json = 'resources/labels.json'
        self.hef_path = 'resources/ai_model.hef'
        self.arch = 'hailo8'
        self.show_fps = True
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        self.post_process_so = os.path.join(os.path.dirname(__file__),
            '../resources/libyolo_hailortpp_postprocess.so')
        self.post_function_name = "filter_letterbox"
        self.create_pipeline()

    def get_pipeline_string(self):
        pipeline_string = self.user_data.get_fallback_pipeline_string()
        return pipeline_string

    def create_pipeline(self):
        Gst.init(None)
        pipeline_string = self.get_pipeline_string()
        self.pipeline = Gst.parse_launch(pipeline_string)
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        for stream_id in [0, 1]:
            identity = self.pipeline.get_by_name(f"identity_callback_{stream_id}")
            if identity:
                pad = identity.get_static_pad("src")
                if pad:
                    callback_data = {"user_data": self.user_data, "stream_id": stream_id}
                    probe_id = pad.add_probe(Gst.PadProbeType.BUFFER, self.app_callback, callback_data)
                    if not hasattr(self, 'probe_ids'):
                        self.probe_ids = {}
                    self.probe_ids[f"stream_{stream_id}"] = (identity, pad, probe_id)
        return True

    def shutdown(self, signum=None, frame=None):
        with self.shutdown_lock:
            if self.shutdown_called:
                return
            self.shutdown_called = True
        print("[PostProcess] Pipeline shutting down...")
        self.user_data.tracking_data.shutdown_event.set()
        self.user_data.shutdown_event.set()
        self.pipeline.set_state(Gst.State.PAUSED)
        GLib.usleep(100000)
        self.pipeline.set_state(Gst.State.READY)
        GLib.usleep(100000)
        self.pipeline.set_state(Gst.State.NULL)
        self.pipeline.get_state(5 * Gst.SECOND)
        cv2.destroyAllWindows()
        GLib.idle_add(self.loop.quit)

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("[PostProcess] End-of-stream - videos fully processed")
            self.shutdown()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
        return True

    def run(self):
        signal.signal(signal.SIGINT, self.shutdown)
        if self.use_frame:
            display_process = multiprocessing.Process(target=display_user_data_frame, args=(self.user_data,))
            display_process.start()
        try:
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("ERROR: Pipeline failed to start!")
                bus = self.pipeline.get_bus()
                msg = bus.timed_pop_filtered(Gst.SECOND, Gst.MessageType.ERROR)
                if msg:
                    err, debug = msg.parse_error()
                    print(f"Pipeline error: {err.message}")
                raise Exception("Pipeline failed to start")
            print("[PostProcess] Pipeline running...")
            self.loop.run()
        except Exception as e:
            print(f"Error in pipeline: {e}")
            raise
        finally:
            self.user_data.tracking_data.shutdown_event.set()
            self.user_data.shutdown_event.set()
            with self.shutdown_lock:
                if not self.shutdown_called:
                    try:
                        self.pipeline.set_state(Gst.State.NULL)
                        self.pipeline.get_state(5 * Gst.SECOND)
                    except:
                        pass
            cv2.destroyAllWindows()
            if self.use_frame and 'display_process' in locals():
                if display_process.is_alive():
                    display_process.terminate()
                    display_process.join(timeout=2)


def get_global_track_id(camera_id, local_track_id, features=None, label=None):
    global global_track_counter, local_to_global_id_map, global_track_labels
    global active_objects_per_camera
    if (camera_id, local_track_id) in local_to_global_id_map:
        return local_to_global_id_map[(camera_id, local_track_id)]
    if not label:
        new_global_id = global_track_counter
        global_track_counter += 1
        local_to_global_id_map[(camera_id, local_track_id)] = new_global_id
        return new_global_id
    other_camera = 1 if camera_id == 0 else 0
    available_matches = []
    if label in active_objects_per_camera[other_camera]:
        for other_local_id, other_global_id in active_objects_per_camera[other_camera][label].items():
            already_matched = False
            for our_local_id, our_global_id in active_objects_per_camera[camera_id][label].items():
                if our_global_id == other_global_id:
                    already_matched = True
                    break
            if not already_matched:
                available_matches.append((other_local_id, other_global_id))
    if available_matches:
        matched_local_id, matched_global_id = available_matches[0]
        local_to_global_id_map[(camera_id, local_track_id)] = matched_global_id
        active_objects_per_camera[camera_id][label][local_track_id] = matched_global_id
        global_track_labels[matched_global_id] = label
        return matched_global_id
    new_global_id = global_track_counter
    global_track_counter += 1
    local_to_global_id_map[(camera_id, local_track_id)] = new_global_id
    active_objects_per_camera[camera_id][label][local_track_id] = new_global_id
    global_track_labels[new_global_id] = label
    return new_global_id


def cleanup_inactive_tracks(camera_id, active_local_track_ids):
    global local_to_global_id_map, active_objects_per_camera
    inactive = []
    for (cam_id, local_id), global_id in local_to_global_id_map.items():
        if cam_id == camera_id and local_id not in active_local_track_ids:
            inactive.append((cam_id, local_id, global_id))
    for cam_id, local_id, global_id in inactive:
        del local_to_global_id_map[(cam_id, local_id)]
        label = global_track_labels.get(global_id)
        if label and label in active_objects_per_camera[cam_id]:
            if local_id in active_objects_per_camera[cam_id][label]:
                del active_objects_per_camera[cam_id][label][local_id]
                if not active_objects_per_camera[cam_id][label]:
                    del active_objects_per_camera[cam_id][label]


def analyze_movement_direction(track_id, center, tracking_data, camera_id, global_id, current_bbox):
    camera_movement_history[camera_id][track_id].appendleft(center)
    bbox_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
    camera_bbox_area_history[camera_id][track_id].appendleft(bbox_area)
    global_movement_history[global_id].appendleft((center, camera_id))
    if len(camera_movement_history[camera_id][track_id]) < 5:
        return None
    if len(camera_bbox_area_history[camera_id][track_id]) >= 5:
        areas = list(camera_bbox_area_history[camera_id][track_id])
        avg_area = sum(areas) / len(areas)
        area_variance = sum((a - avg_area) ** 2 for a in areas) / len(areas)
        area_std_dev = area_variance ** 0.5
        if area_std_dev > (avg_area * 0.8):
            return None
    first_y = camera_movement_history[camera_id][track_id][-1][1]
    last_y = camera_movement_history[camera_id][track_id][0][1]
    if abs(last_y - first_y) < 30:
        return None
    movement_directions = []
    for i in range(1, len(camera_movement_history[camera_id][track_id])):
        curr_y = camera_movement_history[camera_id][track_id][i-1][1]
        prev_y = camera_movement_history[camera_id][track_id][i][1]
        movement_directions.append(1 if curr_y > prev_y else -1)
    positive = sum(1 for d in movement_directions if d > 0)
    negative = sum(1 for d in movement_directions if d < 0)
    if max(positive, negative) / len(movement_directions) < 0.8:
        return None
    total_movement = 0
    for i in range(1, len(camera_movement_history[camera_id][track_id])):
        curr_y = camera_movement_history[camera_id][track_id][i-1][1]
        prev_y = camera_movement_history[camera_id][track_id][i][1]
        total_movement += curr_y - prev_y
    avg_movement = total_movement / 4
    if abs(avg_movement) < 5:
        return None
    current_direction = 'exit' if avg_movement > 0 else 'entry'
    if global_id in global_last_counted_direction:
        if current_direction != global_last_counted_direction[global_id]:
            if global_last_counted_direction[global_id] in tracking_data.counted_tracks:
                tracking_data.counted_tracks[global_last_counted_direction[global_id]].discard(global_id)
    global_last_counted_direction[global_id] = current_direction
    return current_direction


def detection_callback(pad, info, callback_data):
    global camera_covered, cover_alert_thread, blink
    user_data = callback_data["user_data"]
    stream_id = callback_data["stream_id"]
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    format, width, height = get_caps_from_pad(pad)
    if not all([format, width, height]):
        return Gst.PadProbeReturn.OK
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    frame = get_numpy_from_buffer(buffer, format, width, height)
    if is_frame_dark(frame):
        if not camera_covered:
            camera_covered = True
    else:
        if camera_covered:
            camera_covered = False
    active_local_track_ids = set()
    if hasattr(user_data, 'transaction_id') and user_data.transaction_id:
        transaction_memory_manager.track_frame(user_data.transaction_id)
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        class_id = detection.get_class_id()
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
            active_local_track_ids.add(track_id)
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int(bbox.xmax() * width)
        y2 = int(bbox.ymax() * height)
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        global_id = get_global_track_id(stream_id, track_id, None, label)
        if hasattr(user_data, 'transaction_id') and user_data.transaction_id:
            transaction_memory_manager.track_object(user_data.transaction_id, track_id, global_id)
        validation_result = user_data.validate_detected_product(label)
        color = compute_color_for_labels(class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label} L:{track_id} G:{global_id} {'Valid' if validation_result['valid'] else 'Invalid'}"
        text_color = (0, 255, 0) if validation_result['valid'] else (0, 0, 255)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        draw_trail(frame, track_id, center, color, global_id=global_id)
        direction = analyze_movement_direction(track_id, center, user_data.tracking_data, stream_id, global_id, (x1, y1, x2, y2))
        if direction:
            should_count = (
                global_id not in user_data.tracking_data.counted_tracks.get(direction, set()) or
                (global_id in global_last_counted_direction and direction != global_last_counted_direction[global_id])
            )
            if should_count:
                user_data.tracking_data.class_counters[direction][label] += 1
                if direction not in user_data.tracking_data.counted_tracks:
                    user_data.tracking_data.counted_tracks[direction] = set()
                user_data.tracking_data.counted_tracks[direction].add(global_id)
                if validation_result['valid']:
                    if label not in user_data.tracking_data.validated_products[direction]:
                        user_data.tracking_data.validated_products[direction][label] = {
                            "count": 0, "product_details": validation_result['product_details']
                        }
                    user_data.tracking_data.validated_products[direction][label]["count"] += 1
                else:
                    if label not in user_data.tracking_data.invalidated_products[direction]:
                        user_data.tracking_data.invalidated_products[direction][label] = {
                            "count": 0, "raw_detection": {
                                "name": label, "confidence": confidence, "tracking_id": global_id,
                                "bounding_box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}
                            }
                        }
                    user_data.tracking_data.invalidated_products[direction][label]["count"] += 1
    cleanup_inactive_tracks(stream_id, active_local_track_ids)
    user_data.tracking_data.last_time = time.time()
    label = next((det.get_label() for det in detections), None)
    draw_counts(frame, user_data.tracking_data.class_counters, label)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if stream_id == 0:
        user_data.frame_left = frame
    elif stream_id == 1:
        user_data.frame_right = frame
    if hasattr(user_data, "frame_left") and hasattr(user_data, "frame_right"):
        combined_frame = np.hstack((user_data.frame_left, user_data.frame_right))
        user_data.set_frame(combined_frame)
    websocket_data = {
        "validated_products": {
            "entry": {p: {"count": d["count"], "product_details": d["product_details"]}
                      for p, d in user_data.tracking_data.validated_products["entry"].items()},
            "exit": {p: {"count": d["count"], "product_details": d["product_details"]}
                     for p, d in user_data.tracking_data.validated_products["exit"].items()}
        },
        "invalidated_products": {
            "entry": {p: {"count": d["count"], "raw_detection": d["raw_detection"]}
                      for p, d in user_data.tracking_data.invalidated_products["entry"].items()},
            "exit": {p: {"count": d["count"], "raw_detection": d["raw_detection"]}
                     for p, d in user_data.tracking_data.invalidated_products["exit"].items()}
        }
    }
    user_data.tracking_data.websocket_data_manager.update_data(websocket_data)
    return Gst.PadProbeReturn.OK


def reset_tracking_state():
    global global_track_counter, local_to_global_id_map
    global global_movement_history, global_last_counted_direction
    global global_track_labels, active_objects_per_camera
    global object_trails, global_trails
    global camera_movement_history, camera_bbox_area_history
    global_track_counter = 0
    local_to_global_id_map = {}
    global_movement_history = defaultdict(deque)
    global_last_counted_direction = {}
    global_track_labels = {}
    active_objects_per_camera = {0: defaultdict(dict), 1: defaultdict(dict)}
    object_trails = defaultdict(lambda: deque(maxlen=30))
    global_trails = defaultdict(lambda: deque(maxlen=30))
    camera_movement_history = {0: defaultdict(lambda: deque(maxlen=5)), 1: defaultdict(lambda: deque(maxlen=5))}
    camera_bbox_area_history = {0: defaultdict(lambda: deque(maxlen=5)), 1: defaultdict(lambda: deque(maxlen=5))}

async def run_tracking(websocket: WebSocket):
    global readyToProcess, cover_alert_thread
    global unlock_data, done, current_pipeline_app

    unlock_data = 0
    deposit = 0.0
    machine_id = None
    machine_identifier = None
    user_id = None
    transaction_id = None
    product_name = None
    image_count = None
    recorder = None

    while True:
        try:
            message_text = await websocket.receive_text()
            try:
                message = json.loads(message_text)
                if isinstance(message, dict) and message.get('action') == 'start_preview':
                    unlock_data = 1
                    deposit = float(message.get('deposit', 0.0))
                    machine_id = message.get('machine_id')
                    machine_identifier = message.get('machine_identifier')
                    user_id = message.get('user_id')
                    transaction_id = message.get('transaction_id')
                    product_name = message.get('product_name')
                    image_count = message.get('image_count')
                    print(f"Transaction: deposit=${deposit}, id={transaction_id}")
                    break
                else:
                    break
            except json.JSONDecodeError:
                continue
        except Exception as e:
            await websocket.send_json({"status": "error", "message": str(e)})
            return

    if unlock_data == 1:
        readyToProcess = True
        unlock_data = 0

    try:
        if isinstance(message, dict) and message.get('action') == 'product_upload':
            done = True
            machine_id = message.get('machine_id')
            machine_identifier = message.get('machine_identifier')
            user_id = message.get('user_id')
            product_name = message.get('product_name')
            image_count = message.get('image_count')
            alert_dir = "sounds/product_upload_alerts"
            tts_manager.play_mp3_sync(f"{alert_dir}/start_capture.mp3", volume=0.8)
            time.sleep(2)
            camera1_images = capture_images(2, image_count)
            if camera1_images:
                tts_manager.play_mp3_sync(f"{alert_dir}/all_complete.mp3", volume=0.8)
                if upload_images_to_api(camera1_images, machine_id, machine_identifier, user_id, product_name, image_count):
                    delete_images(camera1_images)
                    tts_manager.play_mp3_sync(f"{alert_dir}/upload_success.mp3", volume=0.8)
                else:
                    tts_manager.play_mp3_sync(f"{alert_dir}/upload_failed.mp3", volume=0.8)
            else:
                tts_manager.play_mp3_sync(f"{alert_dir}/upload_failed.mp3", volume=0.8)
        else:
            if transaction_id:
                transaction_memory_manager.start_transaction(transaction_id)

            recorder = DualCameraRecorder(transaction_id=transaction_id, cam0_device=0, cam1_device=2)
            recorder.start_recording()

            await websocket.send_json({"status": "recording", "message": "Cameras recording. Take your items."})

            door_monitor_active = True
            done = True
            while door_monitor_active:
                door_sw = GPIO.input(DOOR_SWITCH_PIN)
                if door_sw == 0:
                    print("[Transaction] Door closed!")
                    door_monitor_active = False
                    break
                await asyncio.sleep(0.1)

            cam0_path, cam1_path = recorder.stop_recording()

            await websocket.send_json({"status": "processing", "message": "Door closed. Analyzing transaction..."})

            reset_tracking_state()

            os.environ['POSTPROCESS_CAM0_VIDEO'] = cam0_path
            os.environ['POSTPROCESS_CAM1_VIDEO'] = cam1_path

            callback = HailoDetectionCallback(websocket, deposit, machine_id, machine_identifier, user_id, transaction_id)

            print(f"[PostProcess] Running pipeline on saved videos...")
            print(f"  Cam0: {cam0_path}")
            print(f"  Cam1: {cam1_path}")

            pipeline_app = HailoDetectionApp(detection_callback, callback)

            with pipeline_lock:
                if current_pipeline_app is not None:
                    current_pipeline_app.shutdown()
                    time.sleep(2)
                current_pipeline_app = pipeline_app

            pipeline_app.run()

            final_data = callback.tracking_data.websocket_data_manager.get_current_data()
            total_price = calculate_total_price_and_control_buzzer(final_data, deposit, None)
            refund = max(0, deposit - total_price)

            results = {
                "status": "completed",
                "message": "Transaction processed",
                "validated_products": final_data.get("validated_products", {}),
                "invalidated_products": final_data.get("invalidated_products", {}),
                "total_price": total_price,
                "deposit": deposit,
                "refund_amount": refund,
            }
            await websocket.send_json(results)

            print(f"RESULTS: Price=${total_price:.2f}, Refund=${refund:.2f}")

            for vpath in [cam0_path, cam1_path]:
                if vpath and os.path.exists(vpath):
                    dname = os.path.splitext(os.path.basename(vpath))[0]
                    threading.Thread(target=stream_video_to_api,
                        args=(vpath, dname, transaction_id, machine_id, user_id, machine_identifier),
                        daemon=True).start()

            if transaction_id:
                transaction_memory_manager.end_transaction(transaction_id)

    except Exception as e:
        print(f"Error: {e}")
        if transaction_id:
            try:
                transaction_memory_manager.end_transaction(transaction_id)
            except:
                pass

    finally:
        try:
            await websocket.send_json({"status": "stopped", "message": "Complete"})
        except:
            pass
        if recorder and recorder.is_recording:
            recorder.stop_recording()
        door_monitor_active = False
        if cover_alert_thread is not None and cover_alert_thread.is_alive():
            camera_covered = False
            tts_manager.stop_all_audio()
            cover_alert_thread.join()
            cover_alert_thread = None
        with pipeline_lock:
            if current_pipeline_app is not None:
                try:
                    with current_pipeline_app.shutdown_lock:
                        if not current_pipeline_app.shutdown_called:
                            current_pipeline_app.shutdown()
                    time.sleep(2)
                except:
                    pass
                finally:
                    current_pipeline_app = None
        cv2.destroyAllWindows()


def setup_product_upload_alerts():
    alert_dir = "sounds/product_upload_alerts"
    os.makedirs(alert_dir, exist_ok=True)
    alerts = {
        "start_capture": "Get ready to capture images. Please prepare your products.",
        "camera_switch": "Switching to the next camera. Please wait.",
        "capture_ready": "Position your product now. Image will be captured shortly.",
        "image_captured": "Image captured successfully.",
        "next_position": "Next position.",
        "upload_success": "All images uploaded successfully. Thank you.",
        "upload_failed": "Upload failed. Please contact support.",
        "all_complete": "Image capture completed. Processing your upload."
    }
    for name, text in alerts.items():
        path = os.path.join(alert_dir, f"{name}.mp3")
        if not os.path.exists(path):
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(path)
    return {name: os.path.join(alert_dir, f"{name}.mp3") for name in alerts}


def capture_images(device_id, num_images=3):
    image_paths = []
    alert_dir = "sounds/product_upload_alerts"
    os.makedirs('camera_images', exist_ok=True)
    try:
        cap = cv2.VideoCapture(device_id)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            return []
        tts_manager.play_mp3_sync(f"{alert_dir}/capture_ready.mp3", volume=0.8)
        for i in range(1, num_images + 1):
            time.sleep(0.5)
            for _ in range(5):
                cap.read()
            ret, frame = cap.read()
            if not ret:
                continue
            filename = os.path.join('camera_images', f"camera_{device_id}_image_{i}.jpg")
            cv2.imwrite(filename, frame)
            image_paths.append(filename)
            tts_manager.play_mp3_sync(f"{alert_dir}/image_captured.mp3", volume=0.8)
            if i < num_images:
                tts_manager.play_mp3_async(f"{alert_dir}/next_position.mp3", volume=0.8)
                time.sleep(1)
        cap.release()
        return image_paths
    except Exception as e:
        print(f"Error with camera {device_id}: {e}")
        if 'cap' in locals():
            cap.release()
        return []


def upload_images_to_api(camera1_images, machine_id, machine_identifier, user_id, product_name, image_count):
    api_url = "https://stg-sfapi.nuboxtech.com/index.php/mobile_app/product/Product/upload_product_images"
    username = 'admin'
    password = '1234'
    api_key = '123456'
    payload = {'machine_id': machine_id, 'machine_identifier': machine_identifier,
               'user_id': user_id, 'product_name': product_name, 'image_count': image_count}
    headers = {'x-api-key': api_key}
    files = []
    opened_files = []
    try:
        for i, img_path in enumerate(camera1_images):
            fh = open(img_path, 'rb')
            opened_files.append(fh)
            files.append(('image[]', (f'camera1_{i}.jpg', fh, 'image/jpeg')))
        response = requests.post(api_url, auth=HTTPBasicAuth(username, password),
                                 headers=headers, data=payload, files=files)
        return response.status_code == 200
    except:
        return False
    finally:
        for fh in opened_files:
            try:
                fh.close()
            except:
                pass


def delete_images(image_paths):
    for path in image_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass


class TransactionMemoryManager:
    def __init__(self):
        self.active_transactions = {}
        self.transaction_history = deque(maxlen=100)
        self.global_stats = {'total_transactions': 0, 'total_cleanups': 0,
                             'average_memory_per_transaction': 0, 'peak_memory': 0}
        self.lock = threading.Lock()

    def start_transaction(self, transaction_id):
        with self.lock:
            process = psutil.Process()
            mem = process.memory_info().rss / 1024 / 1024
            self.active_transactions[transaction_id] = {
                'start_time': time.time(), 'start_memory_mb': mem,
                'tracks_created': set(), 'trails_created': set(), 'frames_processed': 0
            }
            self.global_stats['total_transactions'] += 1
            print(f"[Transaction] Started: {transaction_id} (Memory: {mem:.1f}MB)")

    def end_transaction(self, transaction_id):
        with self.lock:
            if transaction_id not in self.active_transactions:
                return
            trans = self.active_transactions[transaction_id]
            duration = time.time() - trans['start_time']
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            self.transaction_history.append({
                'transaction_id': transaction_id, 'duration': duration,
                'memory_used_mb': mem_before - trans['start_memory_mb'],
                'frames': trans['frames_processed'], 'timestamp': datetime.now()
            })
            del self.active_transactions[transaction_id]
            for _ in range(3):
                if sum(gc.collect(gen) for gen in range(3)) == 0:
                    break
            self.global_stats['total_cleanups'] += 1
            try:
                if sys.platform == 'linux':
                    ctypes.CDLL('libc.so.6').malloc_trim(0)
            except:
                pass
            mem_after = process.memory_info().rss / 1024 / 1024
            print(f"[Transaction] Ended: {transaction_id} ({duration:.1f}s, freed {mem_before - mem_after:.1f}MB)")

    def track_frame(self, transaction_id):
        if transaction_id in self.active_transactions:
            self.active_transactions[transaction_id]['frames_processed'] += 1

    def track_object(self, transaction_id, track_id, global_id):
        if transaction_id in self.active_transactions:
            self.active_transactions[transaction_id]['tracks_created'].add(track_id)
            self.active_transactions[transaction_id]['trails_created'].add(global_id)

    def get_stats(self):
        process = psutil.Process()
        return {
            'current_memory_mb': process.memory_info().rss / 1024 / 1024,
            'active_transactions': len(self.active_transactions),
            'total_transactions': self.global_stats['total_transactions'],
            'total_cleanups': self.global_stats['total_cleanups'],
            'recent_transactions': list(self.transaction_history)[-10:]
        }

    def print_stats(self):
        stats = self.get_stats()
        print(f"Memory: {stats['current_memory_mb']:.1f}MB | "
              f"Transactions: {stats['total_transactions']} | Cleanups: {stats['total_cleanups']}")


transaction_memory_manager = TransactionMemoryManager()


class TTSManager:
    def __init__(self):
        self.tts_lock = Lock()
        self.audio_lock = Lock()
        self.deposit_sounds_dir = "sounds/deposits"
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        except Exception as e:
            print(f"Audio init error: {e}")

    def play_mp3(self, file_path, volume=0.7, wait_for_completion=True):
        def _play():
            with self.audio_lock:
                try:
                    if not os.path.exists(file_path):
                        return False
                    pygame.mixer.music.load(file_path)
                    pygame.mixer.music.set_volume(volume)
                    pygame.mixer.music.play()
                    if wait_for_completion:
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                    return True
                except:
                    return False
        if wait_for_completion:
            return _play()
        else:
            threading.Thread(target=_play, daemon=True).start()
            return True

    def play_mp3_async(self, file_path, volume=0.7):
        return self.play_mp3(file_path, volume, False)

    def play_mp3_sync(self, file_path, volume=0.7):
        return self.play_mp3(file_path, volume, True)

    def stop_all_audio(self):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.stop()
        except:
            pass

    def speak_async(self, text, lang='en'):
        def _speak():
            with self.tts_lock:
                try:
                    tts = gTTS(text=text, lang=lang, slow=False)
                    buf = io.BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    pygame.mixer.music.load(buf)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except:
                    try:
                        subprocess.run(['espeak', '-s', '120', text], check=False, capture_output=True)
                    except:
                        pass
        threading.Thread(target=_speak, daemon=True).start()

    def generate_door_audio_files(self):
        try:
            gTTS(text="Open the door", lang='en', slow=False).save("sounds/door_open.mp3")
            gTTS(text="Door is closing", lang='en', slow=False).save("sounds/door_close.mp3")
        except:
            pass

    def speak_door_open(self):
        self.play_mp3_sync("sounds/door_open.mp3", volume=0.8)

    def speak_door_close(self):
        self.play_mp3_sync("sounds/door_close.mp3", volume=0.8)

    def speak_deposit(self, label):
        try:
            if isinstance(label, str):
                items = [label]
            elif isinstance(label, (list, tuple)):
                items = label
            else:
                items = [str(label)]
            if len(items) == 1:
                text = f"Deposit exceeded. Please return the {items[0]} immediately"
            elif len(items) == 2:
                text = f"Deposit exceeded. Please return the {items[0]} and {items[1]} immediately"
            else:
                joined = ", ".join(items[:-1]) + f", and {items[-1]}"
                text = f"Deposit exceeded. Please return the {joined} immediately"
            self.speak_async(text)
        except:
            pass

    def generate_common_deposit_messages(self):
        pass

    def cleanup(self):
        try:
            self.stop_all_audio()
            pygame.mixer.quit()
        except:
            pass


tts_manager = TTSManager()

done = False

@app.websocket("/ws/track")
async def websocket_endpoint(websocket: WebSocket):
    global readyToProcess, done, current_pipeline_app
    deposit = 0.0
    machine_id = None
    machine_identifier = None
    user_id = None
    transaction_id = None

    await websocket.accept()
    print("WebSocket connection accepted")
    tts_manager.speak_door_open()
    GPIO.output(DOOR_LOCK_PIN, GPIO.LOW)
    GPIO.output(LED_RED, GPIO.LOW)
    GPIO.output(LED_GREEN, GPIO.LOW)

    try:
        start_time = time.time()
        readyToProcess = True
        while readyToProcess and time.time() - start_time < 5:
            door_sw = GPIO.input(DOOR_SWITCH_PIN)
            if door_sw == 1:
                await run_tracking(websocket)
                readyToProcess = False
            else:
                readyToProcess = True
                await asyncio.sleep(0.1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        with pipeline_lock:
            if current_pipeline_app is not None:
                try:
                    with current_pipeline_app.shutdown_lock:
                        if not current_pipeline_app.shutdown_called:
                            current_pipeline_app.shutdown()
                    time.sleep(2)
                except:
                    pass
                finally:
                    current_pipeline_app = None
        await cleanup_websocket_sounds()
        tts_manager.speak_door_close()
        GPIO.output(DOOR_LOCK_PIN, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(LED_GREEN, GPIO.HIGH)
        GPIO.output(LED_RED, GPIO.HIGH)
        if transaction_memory_manager.global_stats['total_transactions'] % 10 == 0:
            transaction_memory_manager.print_stats()
        await websocket.close()


async def cleanup_websocket_sounds():
    global camera_covered_sound_playing, price_alert_sound_playing
    camera_covered_sound_playing = False
    price_alert_sound_playing = False
    tts_manager.stop_all_audio()


@app.get("/health")
async def health_check():
    stats = transaction_memory_manager.get_stats()
    return {
        "status": "healthy" if stats['current_memory_mb'] < 1000 else "warning",
        "memory_mb": round(stats['current_memory_mb'], 2),
        "active_transactions": stats['active_transactions'],
        "total_transactions": stats['total_transactions'],
        "architecture": "post-processing",
        "uptime_hours": round((time.time() - app.start_time) / 3600, 2) if hasattr(app, 'start_time') else 0
    }


@app.get("/stats")
async def get_stats():
    stats = transaction_memory_manager.get_stats()
    return {
        "memory": {"current_mb": round(stats['current_memory_mb'], 2),
                    "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
                    "percent": round(psutil.virtual_memory().percent, 2)},
        "transactions": {"active": stats['active_transactions'],
                         "total": stats['total_transactions'],
                         "cleanups": stats['total_cleanups']},
        "recent": stats['recent_transactions']
    }


def main():
    parser = argparse.ArgumentParser(description='Smart Fridge Post-Processing System')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    app.start_time = time.time()
    os.makedirs('saved_videos', exist_ok=True)
    os.makedirs('camera_images', exist_ok=True)
    os.makedirs("sounds", exist_ok=True)
    os.makedirs("sounds/deposits", exist_ok=True)
    setup_cover_alert_sound()
    setup_product_upload_alerts()
    if not os.path.exists("sounds/door_open.mp3") or not os.path.exists("sounds/door_close.mp3"):
        tts_manager.generate_door_audio_files()
    tts_manager.generate_common_deposit_messages()
    atexit.register(GPIO.cleanup)
    print(f"\n{'='*60}")
    print(f"SMART FRIDGE POST-PROCESSING SYSTEM STARTED")
    print(f"{'='*60}")
    print(f"Architecture: Record -> Door close -> Hailo AI on video -> Results")
    print(f"Same .hef, same .so, same pipeline, same detection_callback")
    print(f"Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
    print(f"WebSocket: ws://{args.host}:{args.port}/ws/track")
    print(f"Health: http://{args.host}:{args.port}/health")
    print(f"{'='*60}\n")
    uvicorn.run("app_server_postprocess:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
