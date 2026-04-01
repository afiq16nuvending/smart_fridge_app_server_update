"""
=====================================================================
SMART FRIDGE OBJECT DETECTION AND TRACKING SYSTEM
=====================================================================

PURPOSE:
This system uses computer vision and AI to track products taken from
a smart refrigerator. It monitors customer transactions, validates
products against inventory, manages door access, and handles payments.

MAIN COMPONENTS:
1. FastAPI WebSocket server for real-time communication
2. Hailo AI accelerator for object detection
3. Multi-camera tracking system
4. GPIO control for door locks, LEDs, and buzzer
5. Transaction memory management
6. Text-to-speech alerts (beep + spoken announcement for every entry and exit)
7. MQTT publishing for connection status (online/offline via LWT) and door status

HARDWARE REQUIREMENTS:
- Raspberry Pi 5 with Hailo-8L AI accelerator
- 2 USB cameras (dev/video0 and dev/video2)
- Door lock mechanism (GPIO 25)
- Door switch sensor (GPIO 26)
- Green LED (GPIO 23)
- Red LED (GPIO 18)
- Buzzer (GPIO 20)

WORKFLOW:
1. Customer opens app → WebSocket connects → MQTT publishes online + door status
2. Deposit deducted → Door unlocks → Cameras start
3. Customer takes/returns items → AI tracks movements
4. Price calculated in real-time → beep + TTS announces every entry and exit
5. Door closes → MQTT publishes door closed → Closing TTS summary plays
6. Video saved → Refund processed → System cleans up → Ready for next customer

AUTHOR: Afiq
VERSION: 2.1
LAST UPDATED: 2025
=====================================================================
"""

# =====================================================================
# IMPORT SECTION
# =====================================================================

# Web Server Framework
from fastapi import FastAPI, WebSocket
import uvicorn

# Computer Vision and Image Processing
import cv2
import numpy as np

# System and File Operations
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
import wave

# Concurrency and Threading
import threading
import asyncio
from threading import Thread, Lock
from multiprocessing import Event
import multiprocessing
import signal

# Data Structures
from collections import deque, defaultdict
from typing import List, Dict, Tuple
import random

# Hardware Control (Raspberry Pi GPIO)
import RPi.GPIO as GPIO
import atexit

# HTTP Requests for API Communication
import requests
from requests.auth import HTTPBasicAuth

# GStreamer for Video Pipeline (Hailo integration)
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject

# Hailo AI Accelerator
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

# Text-to-Speech
from gtts import gTTS
import pygame

# Memory Management and System Monitoring
import gc
import psutil
import sys
import ctypes
from scipy.spatial import distance
import weakref
import setproctitle

# MQTT Client
from mqtt_client import MQTTClient

# =====================================================================
# FASTAPI APP INSTANCE
# =====================================================================

app = FastAPI()

# Data queue for multi-stream processing
data_deque: Dict[int, deque] = {}

# =====================================================================
# MQTT CONFIGURATION
# =====================================================================
# Set the machine ID for this Pi manually for testing.
# In production, this is read automatically from the WebSocket
# start_preview message via os.environ['MACHINE_ID'].

MQTT_MACHINE_ID = "168"    # ← change this to your test machine ID

# Global MQTT client instance — initialised in main(), used everywhere
mqtt_client: MQTTClient = None

# =====================================================================
# GPIO PIN CONFIGURATION
# =====================================================================
# Pin numbers using BCM (Broadcom) numbering scheme

DOOR_LOCK_PIN   = 25   # Controls electromagnetic door lock
DOOR_SWITCH_PIN = 26   # Detects if door is open/closed
LED_GREEN       = 23   # Green status LED
LED_RED         = 18   # Red alert LED
BUZZER_PIN      = 20   # Alert buzzer

# =====================================================================
# GPIO INITIALIZATION
# =====================================================================

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(BUZZER_PIN,      GPIO.OUT, initial=GPIO.HIGH)  # Buzzer OFF (active low)
GPIO.setup(LED_GREEN,       GPIO.OUT, initial=GPIO.HIGH)  # Green LED OFF
GPIO.setup(LED_RED,         GPIO.OUT, initial=GPIO.HIGH)  # Red LED OFF
GPIO.setup(DOOR_LOCK_PIN,   GPIO.OUT, initial=GPIO.HIGH)  # Door LOCKED
GPIO.setup(DOOR_SWITCH_PIN, GPIO.IN)                      # Door sensor (0=closed, 1=open)

# =====================================================================
# GLOBAL STATE FLAGS
# =====================================================================

readyToProcess = False
blink          = False
alert_thread   = None

camera_covered               = False
cover_alert_thread           = None
camera_covered_sound_playing = False
price_alert_sound_playing    = False
last_alerted_label           = None

current_pipeline_app = None
pipeline_lock        = threading.Lock()
print_lock           = threading.Lock()

done        = False
unlock_data = 0

# =====================================================================
# TRACKING DATA STRUCTURES
# =====================================================================

movement_history       = defaultdict(lambda: deque(maxlen=5))
bbox_area_history      = defaultdict(lambda: deque(maxlen=10))
movement_direction     = {}
last_counted_direction = {}

object_trails = defaultdict(lambda: deque(maxlen=30))
global_trails = defaultdict(lambda: deque(maxlen=30))

# =====================================================================
# CROSS-CAMERA TRACKING STRUCTURES
# =====================================================================

global_track_counter          = 0
local_to_global_id_map        = {}
global_movement_history       = defaultdict(deque)
global_last_counted_direction = {}
global_track_labels           = {}

active_objects_per_camera = {
    0: defaultdict(dict),
    1: defaultdict(dict)
}

cross_camera_candidates = defaultdict(list)

camera_movement_history = {
    0: defaultdict(lambda: deque(maxlen=5)),
    1: defaultdict(lambda: deque(maxlen=5))
}

camera_bbox_area_history = {
    0: defaultdict(lambda: deque(maxlen=5)),
    1: defaultdict(lambda: deque(maxlen=5))
}

# =====================================================================
# HARDWARE CONTROL FUNCTIONS
# =====================================================================

def trigger_buzzer(duration=0.5):
    """
    Trigger the buzzer for a specified duration.
    Active LOW: GPIO.LOW = ON, GPIO.HIGH = OFF.

    Args:
        duration (float): Seconds to keep buzzer on.
    """
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)


def blink_led(pin, times, delay):
    """
    Blink an LED a specified number of times.

    Args:
        pin   (int):   GPIO pin number.
        times (int):   Number of blink cycles.
        delay (float): Delay between ON/OFF states in seconds.
    """
    for _ in range(times):
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(delay)


def control_door(pin, action, duration=0.5):
    """
    Control the electromagnetic door lock.
    GPIO.LOW = unlocked, GPIO.HIGH = locked.

    Args:
        pin      (int):   GPIO pin for the door lock.
        action   (str):   'unlock' or 'lock'.
        duration (float): How long to keep door unlocked.
    """
    if action.lower() == 'unlock':
        print("Unlocking door...")
        GPIO.output(pin, GPIO.LOW)
        time.sleep(duration)
        GPIO.output(pin, GPIO.HIGH)
        print("Door locked again")
    elif action.lower() == 'lock':
        print("Locking door...")
        GPIO.output(pin, GPIO.HIGH)
        print("Door locked")
    else:
        print(f"Invalid action '{action}'. Use 'lock' or 'unlock'")

# =====================================================================
# COLOR COMPUTATION FOR VISUALIZATION
# =====================================================================

def compute_color_for_labels(label):
    """
    Compute a unique BGR color for each product class ID.

    Args:
        label (int): Class ID of the detected object.

    Returns:
        tuple: BGR color tuple for OpenCV drawing.
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    if label == 0:
        color = (85, 45, 255)
    elif label == 2:
        color = (222, 82, 175)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

# =====================================================================
# VISUAL TRAIL DRAWING
# =====================================================================

def draw_trail(frame, track_id, center, color, global_id=None):
    """
    Draw movement trail (breadcrumb path) for a tracked object.

    Args:
        frame     (np.ndarray): Video frame to draw on.
        track_id  (int):        Local track ID.
        center    (tuple):      Current center point (x, y).
        color     (tuple):      BGR color for the trail.
        global_id (int):        Global ID across cameras (optional).
    """
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

# =====================================================================
# ON-SCREEN COUNTER DISPLAY
# =====================================================================

def draw_counts(frame, class_counters, label):
    """
    Draw entry/exit counts on the video frame.

    Args:
        frame          (np.ndarray): Video frame to draw on.
        class_counters (dict):       Dict with 'entry' and 'exit' counts.
        label          (str):        Current product label.
    """
    class_names = {
        0:  "",
        1:  "chickenKatsuCurry",
        2:  "dakgangjeongRice",
        3:  "dragonFruit",
        4:  "guava",
        5:  "kimchiFriedRice",
        6:  "kimchiTuna",
        7:  "mango",
        8:  "mangoMilk",
        9:  "pineappleHoney",
        10: "pinkGuava",
    }

    total_entry = sum(class_counters["entry"].values())
    total_exit  = sum(class_counters["exit"].values())

    cv2.putText(frame, f'Total Entry: {total_entry}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Total Exit: {total_exit}', (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    y_offset   = 110
    all_labels = set(class_counters["entry"].keys()) | set(class_counters["exit"].keys())

    for lbl in all_labels:
        entry_count = class_counters["entry"].get(lbl, 0)
        exit_count  = class_counters["exit"].get(lbl, 0)

        class_id = next((k for k, v in class_names.items() if v == lbl), 0)
        color    = compute_color_for_labels(class_id)

        text = f'{lbl} Entry: {entry_count}, Exit: {exit_count}'
        cv2.putText(frame, text, (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 30

# =====================================================================
# DETECTION ZONE VISUALIZATION
# =====================================================================

def draw_zone(frame):
    """
    Draw detection zone overlay on the frame.

    Args:
        frame (np.ndarray): Video frame to draw on.
    """
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
    cv2.putText(frame, "Detection Zone", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# =====================================================================
# ALERT STATE MANAGEMENT
# =====================================================================

def handle_alert_state():
    """
    Blink the red LED continuously while the price exceeds the deposit.
    Runs in a separate daemon thread. Exits when 'blink' flag is False
    or the door closes.
    """
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

# =====================================================================
# PRICE CALCULATION AND ALERT SYSTEM
# =====================================================================

def calculate_total_price_and_control_buzzer(current_data, deposit, label=None):
    """
    Calculate total price of taken items and trigger alerts if deposit exceeded.

    Args:
        current_data (dict):  Current detection data with validated products.
        deposit      (float): Customer's deposit amount.
        label        (str):   Current product label (optional).

    Returns:
        float: Total price of products taken.
    """
    global blink, alert_thread, price_alert_sound_playing, last_alerted_label

    total_product_price = 0
    validated_products  = current_data.get("validated_products", {})
    all_products        = (set(validated_products.get("entry", {}).keys()) |
                           set(validated_products.get("exit", {}).keys()))
    product_prices      = {}

    for product_name in all_products:
        entry_data = validated_products.get("entry", {}).get(product_name, {"count": 0})
        exit_data  = validated_products.get("exit",  {}).get(product_name, {"count": 0})

        entry_count     = entry_data.get("count", 0)
        exit_count      = exit_data.get("count", 0)
        product_details = exit_data.get("product_details") or entry_data.get("product_details")

        if product_details and "product_price" in product_details:
            price_per_unit = float(product_details["product_price"])
            true_count     = max(0, exit_count - entry_count)
            product_total  = true_count * price_per_unit

            if true_count > 0:
                product_prices[product_name] = product_total
                total_product_price += product_total

    if total_product_price > deposit:
        blink = True

        products_to_return = sorted(product_prices.items(), key=lambda x: x[1], reverse=True)
        products_list      = [p[0] for p in products_to_return]
        products_str       = ",".join(products_list)

        if products_list and (not price_alert_sound_playing or last_alerted_label != products_str):
            price_alert_sound_playing = True
            tts_manager.speak_deposit(products_list)
            last_alerted_label = products_str
            print(f"Price alert: ${total_product_price:.2f} > ${deposit:.2f}")
            print(f"Please return: {products_str}")

        if alert_thread is None or not alert_thread.is_alive():
            alert_thread = threading.Thread(target=handle_alert_state, daemon=True)
            alert_thread.start()

    else:
        blink = False
        GPIO.output(LED_RED, GPIO.LOW)

        if price_alert_sound_playing:
            price_alert_sound_playing = False
            last_alerted_label        = None
            if not camera_covered_sound_playing:
                tts_manager.stop_all_audio()
            print("Price within deposit limit - stopping price alert")

    return total_product_price

# =====================================================================
# DOOR STATUS MONITORING
# =====================================================================

def check_door_status():
    """
    Monitor door switch continuously. Returns True when door closes.

    Returns:
        bool: True when door is detected as closed.
    """
    while True:
        door_sw = 1  # TODO: Replace with GPIO.input(DOOR_SWITCH_PIN)
        with print_lock:
            if door_sw == 0:
                print("Door closed - Shutting down preview frames")
                return True
        time.sleep(0.1)

# =====================================================================
# CAMERA COVER DETECTION
# =====================================================================

def is_frame_dark(frame, threshold=40):
    """
    Detect if the camera is covered by checking frame brightness.

    Args:
        frame     (np.ndarray): Input video frame.
        threshold (int):        Brightness threshold (0-255).

    Returns:
        bool: True if frame is abnormally dark.
    """
    global camera_covered_sound_playing

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    avg_brightness = np.mean(gray)
    return avg_brightness < threshold

# =====================================================================
# CAMERA COVER ALERT SETUP
# =====================================================================

def setup_cover_alert_sound():
    """
    Generate and cache the camera-cover warning MP3.

    Returns:
        str: Path to the generated alert MP3 file.
    """
    alert_dir  = "sounds/cover_alerts"
    alert_file = os.path.join(alert_dir, "camera_covered.mp3")

    os.makedirs(alert_dir, exist_ok=True)

    if not os.path.exists(alert_file):
        alert_text = "Dont cover the camera. Please uncover the camera immediately."
        tts = gTTS(text=alert_text, lang='en', slow=False)
        tts.save(alert_file)
        print(f"Cover alert sound saved to {alert_file}")

    return alert_file

# =====================================================================
# BEEP SOUND GENERATOR
# =====================================================================

def generate_beep_file(path="sounds/beep.wav", freq=880, duration=0.12, volume=0.8):
    """
    Generate a short sine-wave beep WAV file and save it to disk.

    Played through pygame before each TTS announcement so the beep and
    voice both come from the same speaker output.

    Only generated once — skipped if the file already exists.

    Args:
        path     (str):   Output file path.
        freq     (int):   Tone frequency in Hz (880 = high A, crisp and clear).
        duration (float): Beep length in seconds (0.12 s is short but audible).
        volume   (float): Amplitude scale 0.0-1.0.
    """
    if os.path.exists(path):
        return

    sample_rate = 22050
    n_samples   = int(sample_rate * duration)
    t           = np.linspace(0, duration, n_samples, endpoint=False)

    # Sine wave with a short fade-out to avoid a hard click at the end
    sine     = np.sin(2 * np.pi * freq * t)
    fade_len = int(n_samples * 0.2)
    fade     = np.ones(n_samples)
    fade[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
    samples  = (sine * fade * volume * 32767).astype(np.int16)

    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)        # mono
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    print(f"Beep sound generated: {path}")

# =====================================================================
# CAMERA COVER ALERT HANDLER
# =====================================================================

def handle_cover_alert():
    """
    Play audio alert repeatedly while camera is covered.
    Runs in a separate daemon thread. Exits when camera is uncovered
    or door closes.
    """
    global camera_covered

    alert_sound = setup_cover_alert_sound()
    print("Camera covered - playing alert sound")

    while camera_covered:
        if GPIO.input(DOOR_SWITCH_PIN) == 0:
            print("Door closed - stopping alert sound")
            break
        tts_manager.play_mp3_async(alert_sound, volume=0.8)
        time.sleep(3.0)

    print("Camera uncovered - stopping alert sound")

# =====================================================================
# VIDEO DISPLAY AND RECORDING SYSTEM
# =====================================================================

def display_user_data_frame(user_data):
    """
    Main video display loop with recording capability.

    Displays live video feed and records to filesystem.
    Monitors door status for shutdown.

    Args:
        user_data: Container with get_frame(), shutdown_event,
                   transaction_id, machine_id, user_id, machine_identifier.
    """
    door_monitor_thread = threading.Thread(target=check_door_status)
    door_monitor_thread.daemon = True
    door_monitor_thread.start()

    transaction_id     = getattr(user_data, 'transaction_id', None)
    machine_id         = getattr(user_data, 'machine_id', None)
    user_id            = getattr(user_data, 'user_id', None)
    machine_identifier = getattr(user_data, 'machine_identifier', None)

    video_dir = os.path.join(os.getcwd(), "saved_videos")
    os.makedirs(video_dir, exist_ok=True)

    fourcc        = cv2.VideoWriter_fourcc(*'XVID')
    output_video  = None
    timestamp     = time.strftime('%Y%m%d_%H%M%S')
    dataset_name  = f"hailo_detection_{timestamp}_{transaction_id}"
    filename      = os.path.join(video_dir, f"{dataset_name}.avi")

    frame_count       = 0
    fps_start_time    = None
    fps_calculated    = False
    actual_fps        = 13.0
    fps_sample_frames = 30

    try:
        while not user_data.shutdown_event.is_set():
            door_sw = 1  # TODO: Replace with GPIO.input(DOOR_SWITCH_PIN)
            frame   = user_data.get_frame()

            if frame is not None:
                if fps_start_time is None:
                    fps_start_time = time.time()

                frame_count += 1

                if not fps_calculated and frame_count >= fps_sample_frames:
                    elapsed_time = time.time() - fps_start_time
                    actual_fps   = frame_count / elapsed_time
                    fps_calculated = True
                    print(f"Detected actual FPS: {actual_fps:.2f}")

                if output_video is None and fps_calculated:
                    height, width = frame.shape[:2]
                    output_video  = cv2.VideoWriter(
                        filename, fourcc, actual_fps, (width, height), isColor=True
                    )
                    print(f"Started recording to: {filename}")

                if output_video is not None:
                    output_video.write(frame.copy())

                cv2.imshow("Hailo Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Error in display loop: {e}")

    finally:
        print("Cleaning up display resources...")

        if output_video is not None:
            output_video.release()
            print(f"Video saved: {filename} ({frame_count} frames)")

        try:
            GPIO.output(DOOR_LOCK_PIN, GPIO.HIGH)
            GPIO.output(LED_GREEN,     GPIO.HIGH)
            GPIO.output(LED_RED,       GPIO.HIGH)
        except Exception as e:
            print(f"Error cleaning up GPIO: {e}")

        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

        user_data.shutdown_event.set()
        print("Display cleanup complete")

# =====================================================================
# VIDEO UPLOAD TO API
# =====================================================================

def stream_video_to_api(video_path, dataset_name, transaction_id,
                        machine_id, user_id, machine_identifier):
    """
    Upload a recorded transaction video to the backend API.

    Args:
        video_path         (str): Local path to the video file.
        dataset_name       (str): Name for the dataset.
        transaction_id     (str): Unique transaction identifier.
        machine_id         (str): Machine identifier.
        user_id            (str): User identifier.
        machine_identifier (str): Machine identifier string.

    Returns:
        bool: True if upload successful, False otherwise.
    """
    # API endpoint URL
    api_url = "https://stg-sfapi.nuboxtech.com/index.php/shopping_app/machine/TransactionDataset/insert_transactionDataset"

    # Authentication credentials
    username = 'admin'
    password = '1234'
    api_key  = '123456'

    # Get current timestamp for database record
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Extract filename from path
    filename = os.path.basename(video_path)

    # Prepare payload (form data)
    payload = {
        'machine_id':       machine_id,
        'created_by':       user_id,
        'dataset_url':      f"assets/video/machine_transaction_dataset/{machine_identifier}/{dataset_name}.avi",
        'dataset_name':     dataset_name,
        'transaction_id':   transaction_id,
        'created_datetime': current_time
    }

    # Prepare headers
    headers = {'x-api-key': api_key}

    print(f"Streaming video to API: {video_path}")
    print(f"Payload: {payload}")

    try:
        with open(video_path, 'rb') as video_file:
            files    = {'video': (filename, video_file, 'video/avi')}
            response = requests.post(
                api_url,
                auth=HTTPBasicAuth(username, password),
                headers=headers,
                data=payload,
                files=files,
                timeout=30.0
            )

            print(f"API Response Status: {response.status_code}")

            if response.status_code == 200:
                print("Video uploaded successfully")
                return True
            else:
                print(f"Upload failed with status: {response.text}")
                return False

    except Exception as e:
        print(f"Error during video streaming: {e}")
        return False

# =====================================================================
# LOCK FILE HELPERS (prevent double-upload)
# =====================================================================

def create_lock_file(video_path):
    """Create a lock file to mark video as being processed."""
    lock_path = video_path + ".lock"
    if os.path.exists(lock_path):
        return False
    try:
        with open(lock_path, 'w') as f:
            f.write(str(time.time()))
        return True
    except Exception:
        return False


def remove_lock_file(video_path):
    """Remove the lock file after processing is complete."""
    lock_path = video_path + ".lock"
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception as e:
        print(f"Error removing lock file: {e}")

# =====================================================================
# VIDEO MONITORING THREAD
# =====================================================================

def monitor_and_send_videos(video_dir, machine_id, machine_identifier, user_id):
    """
    Background thread that watches for completed videos and uploads them.

    Args:
        video_dir          (str): Directory to watch for video files.
        machine_id         (str): Machine identifier for upload metadata.
        machine_identifier (str): Machine name/code.
        user_id            (str): User identifier.
    """
    processed_videos = set()
    processing_lock  = threading.Lock()

    print(f"Video monitoring started for: {video_dir}")

    while True:
        try:
            video_files = glob.glob(os.path.join(video_dir, "*.avi"))

            for video_path in video_files:
                with processing_lock:
                    if video_path in processed_videos:
                        continue

                if not create_lock_file(video_path):
                    continue

                try:
                    if not os.path.exists(video_path):
                        with processing_lock:
                            processed_videos.add(video_path)
                        continue

                    if is_file_complete_enhanced(video_path):
                        filename     = os.path.basename(video_path)
                        dataset_name = filename.replace('.avi', '')

                        try:
                            parts          = dataset_name.split('_')
                            transaction_id = parts[-1] if len(parts) > 2 else None
                        except Exception:
                            transaction_id = None

                        stream_video_to_api(
                            video_path, dataset_name, transaction_id,
                            machine_id, user_id, machine_identifier
                        )

                        try:
                            if os.path.exists(video_path):
                                os.remove(video_path)
                                print(f"Deleted video: {video_path}")
                        except Exception as e:
                            print(f"Error deleting video: {e}")

                        with processing_lock:
                            processed_videos.add(video_path)

                except Exception as e:
                    print(f"Error processing video {video_path}: {e}")

                finally:
                    remove_lock_file(video_path)

            with processing_lock:
                if len(processed_videos) > 100:
                    processed_videos = set(list(processed_videos)[-50:])

        except Exception as e:
            print(f"Error in video monitoring thread: {e}")

        time.sleep(10)

# =====================================================================
# FILE COMPLETION VERIFICATION
# =====================================================================

def is_file_complete_enhanced(file_path, stable_time=5):
    """
    Check that a video file is no longer being written to.

    Args:
        file_path   (str): Path to video file.
        stable_time (int): Seconds the file must be stable.

    Returns:
        bool: True if file is complete and ready for upload.
    """
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

        if initial_stat.st_size != final_stat.st_size or \
           initial_stat.st_mtime != final_stat.st_mtime:
            return False

        try:
            with open(file_path, 'r+b') as f:
                f.seek(0, 2)
                if f.tell() != final_stat.st_size:
                    return False
        except (IOError, OSError):
            return False

        return True

    except Exception as e:
        print(f"Error in completion check: {e}")
        return False

# =====================================================================
# WEBSOCKET DATA MANAGER
# =====================================================================

class WebSocketDataManager:
    """
    Thread-safe manager for real-time detection data updates.
    """

    def __init__(self):
        self.current_data = {
            "validated_products": {
                "entry": {},
                "exit":  {}
            },
            "invalidated_products": {
                "entry": {},
                "exit":  {}
            }
        }
        self._lock = threading.Lock()

    def update_data(self, new_data):
        """Replace current data. Thread-safe."""
        with self._lock:
            self.current_data = new_data

    def get_current_data(self):
        """Return a shallow copy of current data. Thread-safe."""
        with self._lock:
            return self.current_data.copy()

# =====================================================================
# TRACKING DATA CONTAINER
# =====================================================================

class TrackingData:
    """
    Central container for all per-transaction tracking state.
    """

    def __init__(self):
        self.shutdown_event = Event()

        self.validated_products = {
            "entry": {},
            "exit":  {}
        }
        self.invalidated_products = {
            "entry": {},
            "exit":  {}
        }

        self.class_counters = {
            "entry": defaultdict(int),
            "exit":  defaultdict(int)
        }

        self.counted_tracks = {
            "entry": set(),
            "exit":  set()
        }

        self.machine_planogram      = []
        self.hailo_pipeline_string  = ""
        self.frame_rate_calc        = 1
        self.last_time              = time.time()
        self.websocket_data_manager = WebSocketDataManager()

        self.deposit            = 0.0
        self.machine_id         = None
        self.machine_identifier = None
        self.user_id            = None
        self.transaction_id     = None

    def set_transaction_data(self, deposit, machine_id, machine_identifier,
                             user_id, transaction_id):
        """Store transaction metadata."""
        self.deposit            = deposit
        self.machine_id         = machine_id
        self.machine_identifier = machine_identifier
        self.user_id            = user_id
        self.transaction_id     = transaction_id

# =====================================================================
# HAILO DETECTION CALLBACK CLASS
# =====================================================================

class HailoDetectionCallback(app_callback_class):
    """
    Callback class for the Hailo AI detection pipeline.
    """

    def __init__(self, websocket=None, deposit=0.0, machine_id=None,
                 machine_identifier=None, user_id=None, transaction_id=None):
        super().__init__()

        self.tracking_data      = TrackingData()
        self.use_frame          = True
        self.websocket          = websocket
        self.shutdown_event     = Event()

        self.deposit            = deposit
        self.machine_id         = machine_id
        self.machine_identifier = machine_identifier
        self.user_id            = user_id
        self.transaction_id     = transaction_id

        self.tracking_data.set_transaction_data(
            deposit, machine_id, machine_identifier, user_id, transaction_id
        )

        self.video_directory = os.path.join(os.getcwd(), "saved_videos")
        os.makedirs(self.video_directory, exist_ok=True)

        self.store_machine_id_env(machine_id)
        self.load_machine_planogram()

    # -----------------------------------------------------------------
    # MACHINE ID PERSISTENCE
    # -----------------------------------------------------------------

    def store_machine_id_env(self, machine_id):
        if machine_id is not None:
            os.environ['MACHINE_ID'] = str(machine_id)
            print(f"Machine ID {machine_id} stored in environment")

    def load_machine_id_env(self):
        return os.environ.get('MACHINE_ID')

    # -----------------------------------------------------------------
    # PLANOGRAM CACHING
    # -----------------------------------------------------------------

    def is_planogram_valid_for_machine(self, machine_id):
        try:
            stored = os.environ.get('PLANOGRAM_MACHINE_ID')
            return stored == str(machine_id) if stored else False
        except Exception as e:
            print(f"Error checking planogram validity: {e}")
            return False

    def store_planogram_env(self, planogram_data):
        try:
            os.environ['MACHINE_PLANOGRAM'] = json.dumps(planogram_data)
            current_machine_id = self.load_machine_id_env()
            if current_machine_id:
                os.environ['PLANOGRAM_MACHINE_ID'] = str(current_machine_id)
            self.tracking_data.machine_planogram = planogram_data
            print(f"Planogram cached: {len(planogram_data)} products")
        except Exception as e:
            print(f"Error storing planogram: {e}")

    def load_planogram_env(self):
        try:
            planogram_json = os.environ.get('MACHINE_PLANOGRAM')
            if planogram_json:
                return json.loads(planogram_json)
            print("No planogram found in environment")
            return []
        except Exception as e:
            print(f"Error loading planogram: {e}")
            return []

    # -----------------------------------------------------------------
    # PLANOGRAM LOADING
    # -----------------------------------------------------------------

    def load_machine_planogram(self):
        try:
            current_machine_id = self.load_machine_id_env()

            if not current_machine_id:
                existing = self.load_planogram_env()
                if existing:
                    self.tracking_data.machine_planogram = existing
                    print(f"Loaded planogram from env cache: {len(existing)} products")
                else:
                    self.tracking_data.machine_planogram = []
                    print("No planogram available — no machine ID set")
                return

            existing = self.load_planogram_env()
            if existing and self.is_planogram_valid_for_machine(current_machine_id):
                self.tracking_data.machine_planogram = existing
                print(f"Using cached planogram for machine {current_machine_id}: "
                      f"{len(existing)} products")
                self.start_planogram_refresh_thread()
                video_monitor_thread = threading.Thread(
                    target=monitor_and_send_videos,
                    args=(self.video_directory, current_machine_id,
                          self.machine_identifier, self.user_id)
                )
                video_monitor_thread.daemon = True
                video_monitor_thread.start()
                return

            print(f"No valid cache for machine {current_machine_id} — fetching from API")
            self.fetch_and_store_initial_planogram(current_machine_id)

        except Exception as e:
            print(f"Error loading planogram: {e}")
            try:
                existing = self.load_planogram_env()
                self.tracking_data.machine_planogram = existing if existing else []
            except Exception as final_error:
                print(f"Final fallback error: {final_error}")
                self.tracking_data.machine_planogram = []

    def get_fallback_pipeline_string(self):
        """
        Return the hardcoded GStreamer pipeline string.
        NOTE: The GStreamer pipeline is NEVER modified.
        """
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
            "v4l2src device=/dev/video0 name=source_0 ! "
            "image/jpeg, width=640, height=360, framerate=25/1 ! "
            "jpegdec ! "
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
            "v4l2src device=/dev/video2 name=source_2 ! "
            "image/jpeg, width=640, height=360, framerate=25/1 ! "
            "jpegdec ! "
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

    # -----------------------------------------------------------------
    # API PLANOGRAM FETCHING
    # -----------------------------------------------------------------

    def fetch_and_store_initial_planogram(self, machine_id):
        # API authentication credentials
        username = 'admin'
        password = '1234'
        api_key  = '123456'
        headers  = {'x-api-key': api_key}

        # Construct API endpoint URL
        api_endpoint = (f'https://stg-sfapi.nuboxtech.com/index.php/'
                        f'mobile_app/machine/Machine_listing/machine_planogram/{machine_id}')

        video_monitor_thread = threading.Thread(
            target=monitor_and_send_videos,
            args=(self.video_directory, machine_id,
                  self.machine_identifier, self.user_id)
        )
        video_monitor_thread.daemon = True
        video_monitor_thread.start()
        print("Video monitoring thread started")

        self.start_planogram_refresh_thread()

        try:
            api_response = requests.get(
                api_endpoint,
                auth=HTTPBasicAuth(username, password),
                headers=headers
            )

            if api_response.status_code == 200:
                machine_planogram = api_response.json().get('machine_planogram', [])
                self.store_planogram_env(machine_planogram)
                print("Initial planogram fetched and stored:")
                for product in machine_planogram:
                    print(f"  ID: {product['product_library_id']}, "
                          f"Name: {product['product_name']}, "
                          f"Price: {product['product_price']}")
            else:
                print(f"Planogram API request failed: {api_response.status_code}")
                self.tracking_data.machine_planogram = []

        except Exception as e:
            print(f"Error fetching planogram: {e}")
            self.tracking_data.machine_planogram = []

    def start_planogram_refresh_thread(self):
        def refresh_planogram():
            # API credentials
            username = 'admin'
            password = '1234'
            api_key  = '123456'
            headers  = {'x-api-key': api_key}

            while True:
                try:
                    refresh_machine_id = self.load_machine_id_env()
                    if not refresh_machine_id:
                        print("No machine ID for refresh — skipping")
                        time.sleep(1000)
                        continue

                    # Construct API endpoint
                    refresh_endpoint = (f'https://stg-sfapi.nuboxtech.com/index.php/'
                                        f'mobile_app/machine/Machine_listing/'
                                        f'machine_planogram/{refresh_machine_id}')

                    api_response = requests.get(
                        refresh_endpoint,
                        auth=HTTPBasicAuth(username, password),
                        headers=headers
                    )

                    if api_response.status_code == 200:
                        new_planogram     = api_response.json().get('machine_planogram', [])
                        current_planogram = self.load_planogram_env()

                        if new_planogram != current_planogram:
                            self.store_planogram_env(new_planogram)
                            print(f"Planogram refreshed: {len(new_planogram)} products")
                        else:
                            print("Planogram unchanged")
                    else:
                        print(f"Planogram refresh failed: {api_response.status_code}")

                except Exception as e:
                    print(f"Error refreshing planogram: {e}")

                time.sleep(1000)

        refresh_thread = threading.Thread(target=refresh_planogram, daemon=True)
        refresh_thread.start()
        print("Planogram refresh thread started (every 1000 seconds)")

    def get_planogram_from_env(self):
        return self.load_planogram_env()

    # -----------------------------------------------------------------
    # PRODUCT VALIDATION
    # -----------------------------------------------------------------

    def validate_detected_product(self, detected_product):
        current_planogram = self.load_planogram_env()
        if current_planogram:
            self.tracking_data.machine_planogram = current_planogram

        normalized = detected_product.replace(' ', '').lower()

        matches = [
            p for p in self.tracking_data.machine_planogram
            if p.get('product_name', '').replace(' ', '').lower() == normalized
        ]

        if matches:
            return {
                "valid":           True,
                "product_details": matches[0],
                "message":         f"{detected_product} validated — found in planogram"
            }
        else:
            return {
                "valid":           False,
                "product_details": None,
                "message":         f"{detected_product} not in planogram"
            }

# =====================================================================
# HAILO DETECTION APPLICATION CLASS
# =====================================================================

class HailoDetectionApp:
    """
    Manages the GStreamer pipeline and Hailo detection lifecycle.
    NOTE: The GStreamer pipeline string is NEVER modified.
    All detection logic is injected via buffer probes on the existing
    identity elements.
    """

    def __init__(self, app_callback, user_data):
        self.app_callback = app_callback
        self.user_data    = user_data

        self.door_monitor_active = True
        self.door_monitor_thread = threading.Thread(target=self.monitor_door, daemon=True)

        self.shutdown_called = False
        self.shutdown_lock   = threading.Lock()

        self.use_frame          = True
        self.labels_json        = 'resources/labels.json'
        self.hef_path           = 'resources/ai_model.hef'
        self.arch               = 'hailo8'
        self.show_fps           = True
        self.batch_size         = 2
        self.network_width      = 640
        self.network_height     = 640
        self.network_format     = "RGB"

        self.post_process_so = os.path.join(
            os.path.dirname(__file__),
            '../resources/libyolo_hailortpp_postprocess.so'
        )
        self.post_function_name = "filter_letterbox"

        self.create_pipeline()
        self.door_monitor_thread.start()

    def get_pipeline_string(self):
        print("Using fallback pipeline string")
        return self.user_data.get_fallback_pipeline_string()

    def create_pipeline(self):
        Gst.init(None)
        pipeline_string = self.get_pipeline_string()
        self.pipeline   = Gst.parse_launch(pipeline_string)
        self.loop       = GLib.MainLoop()

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        for stream_id in [0, 1]:
            identity = self.pipeline.get_by_name(f"identity_callback_{stream_id}")
            if identity:
                pad = identity.get_static_pad("src")
                if pad:
                    callback_data = {"user_data": self.user_data, "stream_id": stream_id}
                    probe_id      = pad.add_probe(
                        Gst.PadProbeType.BUFFER,
                        self.app_callback,
                        callback_data
                    )
                    if not hasattr(self, 'probe_ids'):
                        self.probe_ids = {}
                    self.probe_ids[f"stream_{stream_id}"] = (identity, pad, probe_id)
                    print(f"Probe attached to identity_callback_{stream_id}")
                else:
                    print(f"Warning: no src pad on identity_callback_{stream_id}")
            else:
                print(f"Warning: identity_callback_{stream_id} not found in pipeline")

        return True

    def monitor_door(self):
        start_time = time.time()
        while self.door_monitor_active:
            door_sw = GPIO.input(DOOR_SWITCH_PIN)
            if door_sw == 0 and time.time() - start_time > 5:
                print("Door closed — initiating shutdown")
                self.shutdown()
                break
            time.sleep(0.1)

    def shutdown(self, signum=None, frame=None):
        with self.shutdown_lock:
            if self.shutdown_called:
                return
            self.shutdown_called = True

        print("Shutting down pipeline…")

        self.door_monitor_active = False
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
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Pipeline error: {err}, {debug}")
            loop.quit()
        return True

    def run(self):
        signal.signal(signal.SIGINT, self.shutdown)

        if self.use_frame:
            display_process = multiprocessing.Process(
                target=display_user_data_frame,
                args=(self.user_data,)
            )
            display_process.start()

        try:
            ret = self.pipeline.set_state(Gst.State.PLAYING)

            if ret == Gst.StateChangeReturn.FAILURE:
                bus = self.pipeline.get_bus()
                msg = bus.timed_pop_filtered(Gst.SECOND, Gst.MessageType.ERROR)
                if msg:
                    err, debug = msg.parse_error()
                    print(f"Pipeline failed: {err.message} | {debug}")
                raise Exception("Pipeline failed to start — camera may be busy")

            print("Pipeline started successfully")
            self.loop.run()

        except Exception as e:
            print(f"Error in pipeline: {e}")
            raise

        finally:
            print("Pipeline run() cleanup…")
            self.user_data.tracking_data.shutdown_event.set()
            self.user_data.shutdown_event.set()

            with self.shutdown_lock:
                if not self.shutdown_called:
                    try:
                        self.pipeline.set_state(Gst.State.NULL)
                        self.pipeline.get_state(5 * Gst.SECOND)
                    except Exception as e:
                        print(f"Error in pipeline cleanup: {e}")

            cv2.destroyAllWindows()

            if self.use_frame and 'display_process' in locals():
                if display_process.is_alive():
                    display_process.terminate()
                    display_process.join(timeout=2)

# =====================================================================
# CROSS-CAMERA TRACKING
# =====================================================================

def get_global_track_id(camera_id, local_track_id, features=None, label=None):
    global global_track_counter, local_to_global_id_map
    global global_track_labels, active_objects_per_camera

    if (camera_id, local_track_id) in local_to_global_id_map:
        return local_to_global_id_map[(camera_id, local_track_id)]

    if not label:
        new_id = global_track_counter
        global_track_counter += 1
        local_to_global_id_map[(camera_id, local_track_id)] = new_id
        return new_id

    other_camera = 1 if camera_id == 0 else 0
    available    = []

    if label in active_objects_per_camera[other_camera]:
        for other_local_id, other_global_id in active_objects_per_camera[other_camera][label].items():
            already_matched = any(
                gid == other_global_id
                for gid in active_objects_per_camera[camera_id][label].values()
            )
            if not already_matched:
                available.append((other_local_id, other_global_id))

    if available:
        _, matched_global_id = available[0]
        local_to_global_id_map[(camera_id, local_track_id)] = matched_global_id
        active_objects_per_camera[camera_id][label][local_track_id] = matched_global_id
        global_track_labels[matched_global_id] = label
        return matched_global_id

    new_id = global_track_counter
    global_track_counter += 1
    local_to_global_id_map[(camera_id, local_track_id)] = new_id
    active_objects_per_camera[camera_id][label][local_track_id] = new_id
    global_track_labels[new_id] = label
    return new_id


def cleanup_inactive_tracks(camera_id, active_local_track_ids):
    global local_to_global_id_map, active_objects_per_camera

    inactive = [
        (cam_id, local_id, gid)
        for (cam_id, local_id), gid in local_to_global_id_map.items()
        if cam_id == camera_id and local_id not in active_local_track_ids
    ]

    for cam_id, local_id, global_id in inactive:
        del local_to_global_id_map[(cam_id, local_id)]
        label = global_track_labels.get(global_id)
        if label and label in active_objects_per_camera[cam_id]:
            if local_id in active_objects_per_camera[cam_id][label]:
                del active_objects_per_camera[cam_id][label][local_id]
                if not active_objects_per_camera[cam_id][label]:
                    del active_objects_per_camera[cam_id][label]

# =====================================================================
# MOVEMENT DIRECTION ANALYSIS
# =====================================================================

def analyze_movement_direction(track_id, center, tracking_data,
                               camera_id, global_id, current_bbox):
    camera_movement_history[camera_id][track_id].appendleft(center)

    bbox_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
    camera_bbox_area_history[camera_id][track_id].appendleft(bbox_area)

    global_movement_history[global_id].appendleft((center, camera_id))

    if len(camera_movement_history[camera_id][track_id]) < 5:
        return None

    # CHECK 1: Bounding-box stability
    if len(camera_bbox_area_history[camera_id][track_id]) >= 5:
        areas    = list(camera_bbox_area_history[camera_id][track_id])
        avg_area = sum(areas) / len(areas)
        variance = sum((a - avg_area) ** 2 for a in areas) / len(areas)
        std_dev  = variance ** 0.5
        if std_dev > avg_area * 0.8:
            return None

    # CHECK 2: Total displacement
    history            = camera_movement_history[camera_id][track_id]
    first_y            = history[-1][1]
    last_y             = history[0][1]
    total_displacement = abs(last_y - first_y)
    if total_displacement < 30:
        return None

    # CHECK 3: Directional consistency
    deltas      = [history[i-1][1] - history[i][1] for i in range(1, len(history))]
    n_positive  = sum(1 for d in deltas if d > 0)
    n_negative  = sum(1 for d in deltas if d < 0)
    consistency = max(n_positive, n_negative) / len(deltas)
    if consistency < 0.8:
        return None

    # CHECK 4: Average movement per frame
    avg_movement = sum(deltas) / len(deltas)
    if abs(avg_movement) < 5:
        return None

    current_direction = 'exit' if avg_movement > 0 else 'entry'

    # Handle direction reversal (customer changes mind)
    if global_id in global_last_counted_direction:
        if current_direction != global_last_counted_direction[global_id]:
            old_dir = global_last_counted_direction[global_id]
            if old_dir in tracking_data.counted_tracks:
                tracking_data.counted_tracks[old_dir].discard(global_id)

    global_last_counted_direction[global_id] = current_direction
    return current_direction

# =====================================================================
# NUMBER TO WORDS HELPER
# =====================================================================

def number_to_words(n):
    """
    Convert an integer (1-20) to its spoken word equivalent.

    Args:
        n (int): Number to convert.

    Returns:
        str: Word representation, or the digit string for numbers > 20.
    """
    words = {
        1: "one",   2: "two",      3: "three",  4: "four",
        5: "five",  6: "six",      7: "seven",  8: "eight",
        9: "nine",  10: "ten",     11: "eleven", 12: "twelve",
        13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
        17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"
    }
    return words.get(n, str(n))

# =====================================================================
# PRODUCT MOVEMENT TTS ANNOUNCER
# =====================================================================

SPEECH_NAMES: Dict[str, str] = {
    # Add friendlier spoken names for product labels here if needed.
    # Example:
    #   "100plus":          "100 plus",
    #   "mangoMilk":        "mango milk",
    #   "kimchiFriedRice":  "kimchi fried rice",
    #   "chickenKatsuCurry":"chicken katsu curry",
    #   "dakgangjeongRice": "dakgangjeong rice",
    #   "pineappleHoney":   "pineapple honey",
    #   "pinkGuava":        "pink guava",
}


class ProductMovementAnnouncer:
    """
    Real-time TTS announcer for product entry and exit events.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def reset(self):
        """Clear state. Call at the start of each new transaction."""
        print("[MovementTTS] Announcer reset for new transaction")

    def _beep_and_speak(self, text: str):
        def _run():
            try:
                beep_path = "sounds/beep.wav"
                if os.path.exists(beep_path):
                    tts_manager.play_mp3_sync(beep_path, volume=0.6)
            except Exception as e:
                print(f"[MovementTTS] Beep error: {e}")
            tts_manager.speak_async(text, lang='en')

        threading.Thread(target=_run, daemon=True).start()

    def on_exit(self, label: str):
        spoken_name = SPEECH_NAMES.get(label, label)
        text        = f"one {spoken_name} removed"
        self._beep_and_speak(text)
        print(f"[MovementTTS] EXIT — '{text}'")

    def on_entry(self, label: str):
        spoken_name = SPEECH_NAMES.get(label, label)
        text        = f"one {spoken_name} returned"
        self._beep_and_speak(text)
        print(f"[MovementTTS] ENTRY — '{text}'")

    def speak_closing_summary(self, class_counters: dict):
        """
        Speak end-of-transaction summary synchronously (blocks until done).
        Called from run_tracking() after speak_door_close().
        """
        try:
            all_labels = (set(class_counters["exit"].keys()) |
                          set(class_counters["entry"].keys()))
            net_items  = {}

            for lbl in all_labels:
                exit_count  = class_counters["exit"].get(lbl, 0)
                entry_count = class_counters["entry"].get(lbl, 0)
                net         = max(0, exit_count - entry_count)
                if net > 0:
                    net_items[lbl] = net

            if not net_items:
                message = "Thank you for visiting. Have a great day!"
            else:
                item_phrases = []
                for lbl, count in net_items.items():
                    spoken_name = SPEECH_NAMES.get(lbl, lbl)
                    count_word  = number_to_words(count)
                    item_phrases.append(f"{count_word} {spoken_name}")

                if len(item_phrases) == 1:
                    items_text = item_phrases[0]
                elif len(item_phrases) == 2:
                    items_text = f"{item_phrases[0]} and {item_phrases[1]}"
                else:
                    items_text = (", ".join(item_phrases[:-1]) +
                                  f", and {item_phrases[-1]}")

                plural  = len(item_phrases) > 1
                message = (
                    f"Thank you for shopping with us. "
                    f"The item{'s' if plural else ''} you have taken "
                    f"{'are' if plural else 'is'} {items_text}. "
                    f"Your refund will be processed shortly."
                )

            print(f"[MovementTTS] CLOSING — '{message}'")

            os.makedirs("sounds/closing", exist_ok=True)
            closing_path = "sounds/closing/closing_summary.mp3"

            tts = gTTS(text=message, lang='en', slow=False)
            tts.save(closing_path)

            self.play_mp3_sync(closing_path, volume=0.8)

        except Exception as e:
            print(f"[MovementTTS] Error in closing summary: {e}")

    def play_mp3_sync(self, file_path, volume=0.8):
        """Play an audio file synchronously through pygame (blocking)."""
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.set_volume(volume)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"[MovementTTS] play_mp3_sync error: {e}")


# Global instance
product_movement_announcer = ProductMovementAnnouncer()

# =====================================================================
# MAIN DETECTION CALLBACK FUNCTION
# =====================================================================

def detection_callback(pad, info, callback_data):
    """
    Process each video frame: detect, track, validate, count, announce,
    and push data to the WebSocket.

    Returns:
        Gst.PadProbeReturn.OK: Always continue pipeline processing.
    """
    global camera_covered, cover_alert_thread, blink

    user_data = callback_data["user_data"]
    stream_id = callback_data["stream_id"]

    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    format, width, height = get_caps_from_pad(pad)
    if not all([format, width, height]):
        return Gst.PadProbeReturn.OK

    # STEP 1: Get AI detections from Hailo
    roi        = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # STEP 2: Get video frame for visualization
    frame = get_numpy_from_buffer(buffer, format, width, height)

    # STEP 3: Camera cover detection
    if is_frame_dark(frame):
        if not camera_covered:
            camera_covered = True
            if cover_alert_thread is None or not cover_alert_thread.is_alive():
                cover_alert_thread = threading.Thread(
                    target=handle_cover_alert, daemon=True
                )
                cover_alert_thread.start()
    else:
        if camera_covered:
            camera_covered = False

    # STEP 4: Track active objects for cleanup
    active_local_track_ids = set()

    if hasattr(user_data, 'transaction_id') and user_data.transaction_id:
        transaction_memory_manager.track_frame(user_data.transaction_id)

    # STEP 5: Process each detection
    for detection in detections:
        label      = detection.get_label()
        bbox       = detection.get_bbox()
        confidence = detection.get_confidence()
        class_id   = detection.get_class_id()

        track_id = 0
        track    = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
            active_local_track_ids.add(track_id)

        x1     = int(bbox.xmin() * width)
        y1     = int(bbox.ymin() * height)
        x2     = int(bbox.xmax() * width)
        y2     = int(bbox.ymax() * height)
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        global_id = get_global_track_id(stream_id, track_id, None, label)

        if hasattr(user_data, 'transaction_id') and user_data.transaction_id:
            transaction_memory_manager.track_object(
                user_data.transaction_id, track_id, global_id
            )

        validation_result = user_data.validate_detected_product(label)

        color = compute_color_for_labels(class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = (f"{label} L:{track_id} G:{global_id} "
                      f"{'Valid' if validation_result['valid'] else 'Invalid'}")
        text_color = (0, 255, 0) if validation_result['valid'] else (0, 0, 255)
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        draw_trail(frame, track_id, center, color, global_id=global_id)

        direction = analyze_movement_direction(
            track_id, center, user_data.tracking_data,
            stream_id, global_id, (x1, y1, x2, y2)
        )

        # STEP 6: Update counters and announce
        if direction:
            should_count = (
                global_id not in user_data.tracking_data.counted_tracks.get(direction, set()) or
                (global_id in global_last_counted_direction and
                 direction != global_last_counted_direction[global_id])
            )

            if should_count:
                user_data.tracking_data.class_counters[direction][label] += 1

                if direction == "exit":
                    product_movement_announcer.on_exit(label)
                else:
                    product_movement_announcer.on_entry(label)

                if direction not in user_data.tracking_data.counted_tracks:
                    user_data.tracking_data.counted_tracks[direction] = set()
                user_data.tracking_data.counted_tracks[direction].add(global_id)

                if validation_result['valid']:
                    if label not in user_data.tracking_data.validated_products[direction]:
                        user_data.tracking_data.validated_products[direction][label] = {
                            "count":           0,
                            "product_details": validation_result['product_details']
                        }
                    user_data.tracking_data.validated_products[direction][label]["count"] += 1
                else:
                    if label not in user_data.tracking_data.invalidated_products[direction]:
                        user_data.tracking_data.invalidated_products[direction][label] = {
                            "count":         0,
                            "raw_detection": {
                                "name":         label,
                                "confidence":   confidence,
                                "tracking_id":  global_id,
                                "bounding_box": {"xmin": x1, "ymin": y1,
                                                 "xmax": x2, "ymax": y2}
                            }
                        }
                    user_data.tracking_data.invalidated_products[direction][label]["count"] += 1

    # STEP 7: Cleanup inactive tracks
    cleanup_inactive_tracks(stream_id, active_local_track_ids)

    # STEP 8: Update FPS timestamp
    user_data.tracking_data.last_time = time.time()

    # STEP 9: Draw on-screen counters
    current_label = next((det.get_label() for det in detections), None)
    draw_counts(frame, user_data.tracking_data.class_counters, current_label)

    # STEP 10: Convert RGB → BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # STEP 11: Store frame per camera and combine
    if stream_id == 0:
        user_data.frame_left = frame
    elif stream_id == 1:
        user_data.frame_right = frame

    if hasattr(user_data, "frame_left") and hasattr(user_data, "frame_right"):
        combined_frame = np.hstack((user_data.frame_left, user_data.frame_right))
        user_data.set_frame(combined_frame)

    # STEP 12: Update WebSocket data
    websocket_data = {
        "validated_products": {
            "entry": {
                product: {"count": d["count"], "product_details": d["product_details"]}
                for product, d in user_data.tracking_data.validated_products["entry"].items()
            },
            "exit": {
                product: {"count": d["count"], "product_details": d["product_details"]}
                for product, d in user_data.tracking_data.validated_products["exit"].items()
            }
        },
        "invalidated_products": {
            "entry": {
                product: {"count": d["count"], "raw_detection": d["raw_detection"]}
                for product, d in user_data.tracking_data.invalidated_products["entry"].items()
            },
            "exit": {
                product: {"count": d["count"], "raw_detection": d["raw_detection"]}
                for product, d in user_data.tracking_data.invalidated_products["exit"].items()
            }
        }
    }
    user_data.tracking_data.websocket_data_manager.update_data(websocket_data)

    # STEP 13: Price check and deposit alert
    current_data = user_data.tracking_data.websocket_data_manager.get_current_data()
    calculate_total_price_and_control_buzzer(current_data, user_data.deposit, current_label)

    return Gst.PadProbeReturn.OK

# =====================================================================
# TRANSACTION ORCHESTRATION
# =====================================================================

async def run_tracking(websocket: WebSocket):
    """
    Orchestrate the full lifecycle of one customer transaction.
    """
    global unlock_data, done, current_pipeline_app

    unlock_data        = 0
    deposit            = 0.0
    machine_id         = None
    machine_identifier = None
    user_id            = None
    transaction_id     = None
    product_name       = None
    image_count        = None

    # PHASE 1: Wait for start message
    while True:
        try:
            message_text = await websocket.receive_text()
            print(f"Received: {message_text}")

            try:
                message = json.loads(message_text)

                if isinstance(message, dict) and message.get('action') == 'start_preview':
                    unlock_data        = 1
                    deposit            = float(message.get('deposit', 0.0))
                    machine_id         = message.get('machine_id')
                    machine_identifier = message.get('machine_identifier')
                    user_id            = message.get('user_id')
                    transaction_id     = message.get('transaction_id')
                    product_name       = message.get('product_name')
                    image_count        = message.get('image_count')

                    print(f"Transaction started — deposit: ${deposit}, "
                          f"machine: {machine_id}, user: {user_id}, "
                          f"tx: {transaction_id}")

                    # --------------------------------------------------
                    # MQTT: connect now that machine_id is confirmed.
                    # store_machine_id_env() will be called inside
                    # HailoDetectionCallback, so os.environ['MACHINE_ID']
                    # is set before mqtt_client.connect() resolves topics.
                    # We set it here explicitly for the MQTT LWT topic.
                    # --------------------------------------------------
                    if machine_id:
                        os.environ['MACHINE_ID'] = str(machine_id)
                    if mqtt_client is not None:
                        mqtt_client.connect()
                    # --------------------------------------------------

                    break
                else:
                    break

            except json.JSONDecodeError:
                continue

        except Exception as e:
            await websocket.send_json({
                "status":  "error",
                "message": f"Error waiting for start message: {e}"
            })
            return

    # PHASE 2: Door control
    if unlock_data == 1:
        print("Unlocking door for 0.5 seconds")
        readyToProcess = True
        unlock_data    = 0

    try:
        # MODE 1: Product Upload
        if isinstance(message, dict) and message.get('action') == 'product_upload':
            done               = True
            machine_id         = message.get('machine_id')
            machine_identifier = message.get('machine_identifier')
            user_id            = message.get('user_id')
            product_name       = message.get('product_name')
            image_count        = message.get('image_count')

            print(f"\nProduct Upload Mode: {product_name} ({image_count} images)")

            alert_dir = "sounds/product_upload_alerts"
            tts_manager.play_mp3_sync(f"{alert_dir}/start_capture.mp3", volume=0.8)
            time.sleep(2)

            camera1_images = capture_images(2, image_count)

            if camera1_images:
                tts_manager.play_mp3_sync(f"{alert_dir}/all_complete.mp3", volume=0.8)
                if upload_images_to_api(camera1_images, machine_id, machine_identifier,
                                        user_id, product_name, image_count):
                    delete_images(camera1_images)
                    tts_manager.play_mp3_sync(f"{alert_dir}/upload_success.mp3", volume=0.8)
                else:
                    tts_manager.play_mp3_sync(f"{alert_dir}/upload_failed.mp3", volume=0.8)
            else:
                tts_manager.play_mp3_sync(f"{alert_dir}/upload_failed.mp3", volume=0.8)

        # MODE 2: Detection / Transaction
        else:
            if transaction_id:
                transaction_memory_manager.start_transaction(transaction_id)
                print(f"[Memory] Transaction {transaction_id} started")

            # Reset movement announcer for this transaction
            product_movement_announcer.reset()

            door_monitor_active = True
            done                = True

            async def monitor_door():
                nonlocal door_monitor_active
                while door_monitor_active:
                    door_sw = 1  # TODO: Replace with GPIO.input(DOOR_SWITCH_PIN)
                    if door_sw == 0:
                        print("Door closed — stopping tracking")
                        callback.tracking_data.shutdown_event.set()
                        callback.shutdown_event.set()
                        door_monitor_active = False
                        try:
                            await websocket.send_json({
                                "status":  "stopped",
                                "message": "Door closed — tracking stopped"
                            })
                        except Exception as e:
                            print(f"Error sending door-close message: {e}")
                        break
                    await asyncio.sleep(0.1)

            door_monitor_task = asyncio.create_task(monitor_door())

            callback = HailoDetectionCallback(
                websocket, deposit, machine_id, machine_identifier,
                user_id, transaction_id
            )

            def send_websocket_data():
                while not callback.tracking_data.shutdown_event.is_set():
                    try:
                        current_data = callback.tracking_data.websocket_data_manager.get_current_data()
                        asyncio.run(websocket.send_json(current_data))
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error sending WebSocket data: {e}")

            websocket_sender = threading.Thread(target=send_websocket_data)
            websocket_sender.start()

            def signal_handler(signum, frame):
                print("\nCtrl+C detected — initiating shutdown…")
                callback.tracking_data.shutdown_event.set()
                callback.shutdown_event.set()
                cv2.destroyAllWindows()

            signal.signal(signal.SIGINT, signal_handler)

            detection_app = HailoDetectionApp(detection_callback, callback)

            with pipeline_lock:
                if current_pipeline_app is not None:
                    print("Stopping previous pipeline app…")
                    current_pipeline_app.shutdown()
                    time.sleep(2)
                current_pipeline_app = detection_app

            detection_app.run()

            # ---------------------------------------------------------
            # DOOR CLOSE TTS + CLOSING SUMMARY
            # Both called here inside run_tracking, before the websocket
            # finally block, so cleanup_websocket_sounds() cannot
            # call stop_all_audio() and cut off the audio.
            # speak_door_close() blocks until done, then
            # speak_closing_summary() generates and plays the summary.
            # ---------------------------------------------------------
            tts_manager.speak_door_close()

            if 'callback' in locals() and hasattr(callback, 'tracking_data'):
                product_movement_announcer.speak_closing_summary(
                    callback.tracking_data.class_counters
                )
            # ---------------------------------------------------------

            if transaction_id:
                transaction_memory_manager.end_transaction(transaction_id)
                print(f"[Memory] Transaction {transaction_id} ended")

    except Exception as e:
        print(f"Error during tracking: {e}")
        if transaction_id:
            try:
                transaction_memory_manager.end_transaction(transaction_id)
            except Exception:
                pass

    finally:
        await websocket.send_json({
            "status":  "stopped",
            "message": "Tracking has been fully stopped"
        })

        door_monitor_active = False

        if cover_alert_thread is not None and cover_alert_thread.is_alive():
            camera_covered = False
            tts_manager.stop_all_audio()
            cover_alert_thread.join()

        if 'door_monitor_task' in locals():
            await door_monitor_task
        if 'callback' in locals():
            callback.tracking_data.shutdown_event.set()
            callback.shutdown_event.set()
        if 'websocket_sender' in locals():
            websocket_sender.join()
        if 'detection_app' in locals():
            detection_app.pipeline.set_state(Gst.State.NULL)

        # --------------------------------------------------------------
        # MQTT: disconnect cleanly after tracking ends.
        # This publishes "offline" explicitly before the LWT fires.
        # --------------------------------------------------------------
        if mqtt_client is not None:
            mqtt_client.disconnect()
        # --------------------------------------------------------------

        cv2.destroyAllWindows()

# =====================================================================
# PRODUCT CAPTURE SYSTEM
# =====================================================================

def setup_product_upload_alerts():
    alert_dir = "sounds/product_upload_alerts"
    os.makedirs(alert_dir, exist_ok=True)

    alerts = {
        "start_capture":  "Get ready to capture images. Please prepare your products.",
        "camera_switch":  "Switching to the next camera. Please wait.",
        "capture_ready":  "Position your product now. Image will be captured shortly.",
        "image_captured": "Image captured successfully.",
        "next_position":  "Next position.",
        "upload_success": "All images uploaded successfully. Thank you.",
        "upload_failed":  "Upload failed. Please contact support.",
        "all_complete":   "Image capture completed. Processing your upload."
    }

    generated = {}
    for name, text in alerts.items():
        path = os.path.join(alert_dir, f"{name}.mp3")
        if not os.path.exists(path):
            gTTS(text=text, lang='en', slow=False).save(path)
            print(f"Generated: {path}")
        generated[name] = path

    print(f"Product upload alerts ready in {alert_dir}")
    return generated


def capture_images(device_id, num_images=3):
    image_paths = []
    alert_dir   = "sounds/product_upload_alerts"
    os.makedirs('camera_images', exist_ok=True)

    try:
        cap = cv2.VideoCapture(device_id)
        cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not cap.isOpened():
            print(f"Error: Could not open camera {device_id}")
            return []

        tts_manager.play_mp3_sync(f"{alert_dir}/capture_ready.mp3", volume=0.8)

        for i in range(1, num_images + 1):
            time.sleep(0.5)
            for _ in range(5):
                cap.read()

            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture image {i} from camera {device_id}")
                continue

            filename = os.path.join('camera_images', f"camera_{device_id}_image_{i}.jpg")
            cv2.imwrite(filename, frame)
            image_paths.append(filename)
            print(f"Saved {filename}")

            tts_manager.play_mp3_sync(f"{alert_dir}/image_captured.mp3", volume=0.8)

            if i < num_images:
                tts_manager.play_mp3_async(f"{alert_dir}/next_position.mp3", volume=0.8)
                time.sleep(1)

        cap.release()
        return image_paths

    except Exception as e:
        print(f"Error capturing from camera {device_id}: {e}")
        if 'cap' in locals():
            cap.release()
        return []


def upload_images_to_api(camera1_images, machine_id, machine_identifier,
                         user_id, product_name, image_count):
    api_url = "https://stg-sfapi.nuboxtech.com/index.php/mobile_app/product/Product/upload_product_images"

    username = 'admin'
    password = '1234'
    api_key  = '123456'

    headers = {'x-api-key': api_key}
    payload = {
        'machine_id':         machine_id,
        'machine_identifier': machine_identifier,
        'user_id':            user_id,
        'product_name':       product_name,
        'image_count':        image_count
    }

    files        = []
    opened_files = []

    try:
        for i, img_path in enumerate(camera1_images):
            fh = open(img_path, 'rb')
            opened_files.append(fh)
            files.append(('image[]', (f'camera1_{i}.jpg', fh, 'image/jpeg')))

        response = requests.post(
            api_url,
            auth=HTTPBasicAuth(username, password),
            headers=headers,
            data=payload,
            files=files
        )

        print(f"Image upload response: {response.status_code}")
        return response.status_code == 200

    except Exception as e:
        print(f"Error uploading images: {e}")
        return False

    finally:
        for fh in opened_files:
            try:
                fh.close()
            except Exception:
                pass


def delete_images(image_paths):
    deleted = 0
    for path in image_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted: {path}")
                deleted += 1
        except Exception as e:
            print(f"Error deleting {path}: {e}")
    print(f"Deleted {deleted} image(s)")
    return deleted

# =====================================================================
# TRANSACTION-BASED MEMORY MANAGEMENT
# =====================================================================

class TransactionMemoryManager:

    def __init__(self):
        self.active_transactions = {}
        self.transaction_history = deque(maxlen=100)
        self.global_stats = {
            'total_transactions': 0,
            'total_cleanups':     0,
            'peak_memory':        0
        }
        self.lock = threading.Lock()

    def start_transaction(self, transaction_id):
        with self.lock:
            process      = psutil.Process()
            memory_start = process.memory_info().rss / 1024 / 1024
            self.active_transactions[transaction_id] = {
                'start_time':       time.time(),
                'start_memory_mb':  memory_start,
                'tracks_created':   set(),
                'trails_created':   set(),
                'frames_processed': 0
            }
            self.global_stats['total_transactions'] += 1
            print(f"[Transaction] {transaction_id} started | memory: {memory_start:.1f}MB")

    def end_transaction(self, transaction_id):
        with self.lock:
            if transaction_id not in self.active_transactions:
                print(f"[Transaction] Warning: {transaction_id} not found")
                return

            trans_data = self.active_transactions[transaction_id]
            duration   = time.time() - trans_data['start_time']
            process    = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            mem_used   = mem_before - trans_data['start_memory_mb']

            print(f"[Transaction] {transaction_id} ending | "
                  f"duration: {duration:.1f}s | memory used: {mem_used:.1f}MB")

            self.transaction_history.append({
                'transaction_id': transaction_id,
                'duration':       duration,
                'memory_used_mb': mem_used,
                'frames':         trans_data['frames_processed'],
                'timestamp':      datetime.now()
            })

            del self.active_transactions[transaction_id]

            self._cleanup_transaction_data(transaction_id, trans_data)
            self._recreate_global_dictionaries()
            self._aggressive_garbage_collection()
            self._release_memory_to_os()

            mem_after = process.memory_info().rss / 1024 / 1024
            mem_freed = mem_before - mem_after
            print(f"[Transaction] Freed {mem_freed:.1f}MB | memory now: {mem_after:.1f}MB")

    def _cleanup_transaction_data(self, transaction_id, trans_data):
        global object_trails, global_trails, camera_movement_history
        global camera_bbox_area_history, local_to_global_id_map
        global active_objects_per_camera, global_movement_history

        tracks = trans_data['tracks_created']
        trails = trans_data['trails_created']

        for tid in list(object_trails.keys()):
            if tid in tracks:
                object_trails[tid].clear()
                del object_trails[tid]

        for gid in list(global_trails.keys()):
            if gid in trails:
                global_trails[gid].clear()
                del global_trails[gid]

        for cam_id in [0, 1]:
            for tid in list(camera_movement_history[cam_id].keys()):
                if tid in tracks:
                    camera_movement_history[cam_id][tid].clear()
                    del camera_movement_history[cam_id][tid]
            for tid in list(camera_bbox_area_history[cam_id].keys()):
                if tid in tracks:
                    camera_bbox_area_history[cam_id][tid].clear()
                    del camera_bbox_area_history[cam_id][tid]

        for key in list(local_to_global_id_map.keys()):
            _, local_id = key
            if local_id in tracks:
                del local_to_global_id_map[key]

        for cam_id in [0, 1]:
            for lbl in list(active_objects_per_camera[cam_id].keys()):
                for lid in list(active_objects_per_camera[cam_id][lbl].keys()):
                    if lid in tracks:
                        del active_objects_per_camera[cam_id][lbl][lid]
                if not active_objects_per_camera[cam_id][lbl]:
                    del active_objects_per_camera[cam_id][lbl]

        for gid in list(global_movement_history.keys()):
            if gid in trails:
                global_movement_history[gid].clear()
                del global_movement_history[gid]

    def _recreate_global_dictionaries(self):
        global object_trails, global_trails, camera_movement_history
        global camera_bbox_area_history, active_objects_per_camera
        global global_movement_history, local_to_global_id_map

        print("[Cleanup] Recreating global dictionaries…")

        active_tracks     = set()
        active_trails_set = set()
        for td in self.active_transactions.values():
            active_tracks.update(td['tracks_created'])
            active_trails_set.update(td['trails_created'])

        new_obj_trails = defaultdict(lambda: deque(maxlen=30))
        for tid, trail in object_trails.items():
            if tid in active_tracks:
                new_obj_trails[tid] = trail
        object_trails = new_obj_trails

        new_global_trails = defaultdict(lambda: deque(maxlen=30))
        for gid, trail in global_trails.items():
            if gid in active_trails_set:
                new_global_trails[gid] = trail
        global_trails = new_global_trails

        new_cam_hist = {
            0: defaultdict(lambda: deque(maxlen=5)),
            1: defaultdict(lambda: deque(maxlen=5))
        }
        for cam_id in [0, 1]:
            for tid, hist in camera_movement_history[cam_id].items():
                if tid in active_tracks:
                    new_cam_hist[cam_id][tid] = hist
        camera_movement_history = new_cam_hist

        new_bbox_hist = {
            0: defaultdict(lambda: deque(maxlen=5)),
            1: defaultdict(lambda: deque(maxlen=5))
        }
        for cam_id in [0, 1]:
            for tid, hist in camera_bbox_area_history[cam_id].items():
                if tid in active_tracks:
                    new_bbox_hist[cam_id][tid] = hist
        camera_bbox_area_history = new_bbox_hist

        print("[Cleanup] Dictionary recreation complete")

    def _aggressive_garbage_collection(self):
        total = 0
        for pass_num in range(3):
            collected = [gc.collect(gen) for gen in range(3)]
            total    += sum(collected)
            print(f"[GC] Pass {pass_num + 1}: {collected} (total={sum(collected)})")
            if sum(collected) == 0:
                break
        print(f"[GC] Total collected: {total}")
        self.global_stats['total_cleanups'] += 1

    def _release_memory_to_os(self):
        try:
            if sys.platform == 'linux':
                ctypes.CDLL('libc.so.6').malloc_trim(0)
                print("[Memory] malloc_trim() called")
        except Exception as e:
            print(f"[Memory] malloc_trim not available: {e}")

    def track_frame(self, transaction_id):
        if transaction_id in self.active_transactions:
            self.active_transactions[transaction_id]['frames_processed'] += 1

    def track_object(self, transaction_id, track_id, global_id):
        if transaction_id in self.active_transactions:
            self.active_transactions[transaction_id]['tracks_created'].add(track_id)
            self.active_transactions[transaction_id]['trails_created'].add(global_id)

    def get_stats(self):
        process     = psutil.Process()
        memory_info = process.memory_info()
        return {
            'current_memory_mb':   memory_info.rss / 1024 / 1024,
            'active_transactions': len(self.active_transactions),
            'total_transactions':  self.global_stats['total_transactions'],
            'total_cleanups':      self.global_stats['total_cleanups'],
            'recent_transactions': list(self.transaction_history)[-10:]
        }

    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print("TRANSACTION MEMORY STATISTICS")
        print("="*60)
        print(f"Current Memory:       {stats['current_memory_mb']:.1f}MB")
        print(f"Active Transactions:  {stats['active_transactions']}")
        print(f"Total Transactions:   {stats['total_transactions']}")
        print(f"Total Cleanups:       {stats['total_cleanups']}")
        print("\nRecent Transactions:")
        for t in stats['recent_transactions']:
            print(f"  {t['transaction_id']}: {t['duration']:.1f}s, "
                  f"{t['memory_used_mb']:.1f}MB, {t['frames']} frames")
        print("="*60 + "\n")


# Global instance
transaction_memory_manager = TransactionMemoryManager()

# =====================================================================
# TEXT-TO-SPEECH (TTS) MANAGER
# =====================================================================

class TTSManager:

    def __init__(self):
        self.tts_lock           = Lock()
        self.audio_lock         = Lock()
        self.deposit_sounds_dir = "sounds/deposits"
        self.init_audio_player()

    def init_audio_player(self):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            print("Audio player initialised")
        except Exception as e:
            print(f"Error initialising audio player: {e}")

    def play_mp3(self, file_path, volume=0.7, wait_for_completion=True):
        def _play():
            with self.audio_lock:
                try:
                    if not os.path.exists(file_path):
                        print(f"MP3 not found: {file_path}")
                        return False
                    if not file_path.lower().endswith('.mp3') and \
                       not file_path.lower().endswith('.wav'):
                        print(f"Unsupported format: {file_path}")
                        return False

                    pygame.mixer.music.load(file_path)
                    pygame.mixer.music.set_volume(volume)
                    pygame.mixer.music.play()

                    if wait_for_completion:
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                    return True

                except Exception as e:
                    print(f"Error playing {file_path}: {e}")
                    return False

        if wait_for_completion:
            return _play()
        else:
            threading.Thread(target=_play, daemon=True).start()
            return True

    def play_mp3_async(self, file_path, volume=0.7):
        return self.play_mp3(file_path, volume, wait_for_completion=False)

    def play_mp3_sync(self, file_path, volume=0.7):
        return self.play_mp3(file_path, volume, wait_for_completion=True)

    def play_sound_effect(self, file_path, volume=0.7):
        try:
            if not os.path.exists(file_path):
                return False
            sound = pygame.mixer.Sound(file_path)
            sound.set_volume(volume)
            sound.play()
            return True
        except Exception as e:
            print(f"Error playing sound effect: {e}")
            return False

    def stop_all_audio(self):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.stop()
        except Exception as e:
            print(f"Error stopping audio: {e}")

    def pause_audio(self):
        try:
            pygame.mixer.music.pause()
        except Exception as e:
            print(f"Error pausing audio: {e}")

    def resume_audio(self):
        try:
            pygame.mixer.music.unpause()
        except Exception as e:
            print(f"Error resuming audio: {e}")

    def set_volume(self, volume):
        try:
            pygame.mixer.music.set_volume(volume)
        except Exception as e:
            print(f"Error setting volume: {e}")

    def is_audio_playing(self):
        try:
            return pygame.mixer.music.get_busy()
        except Exception:
            return False

    def generate_common_deposit_messages(self):
        try:
            common_products = [
                "100plus", "coconut", "mineral",
                "water bottle", "energy drink"
            ]
            for product in common_products:
                self.generate_deposit_audio_file(product)

            common_combos = [
                ["coke", "pepsi"],
                ["sprite", "water bottle"],
            ]
            for combo in common_combos:
                self.generate_deposit_audio_file(combo)

            print("Common deposit messages generated")

        except Exception as e:
            print(f"Error generating deposit messages: {e}")

    def generate_deposit_audio_file(self, label):
        try:
            if isinstance(label, str) and "," in label:
                return self.generate_deposit_audio_file(
                    [item.strip() for item in label.split(",")]
                )

            if isinstance(label, str):
                text = f"Deposit exceeded. Please return the {label} immediately"
            elif isinstance(label, (list, tuple)):
                if len(label) == 0:
                    return None
                elif len(label) == 1:
                    text = f"Deposit exceeded. Please return the {label[0]} immediately"
                elif len(label) == 2:
                    text = (f"Deposit exceeded. Please return the "
                            f"{label[0]} and {label[1]} immediately")
                else:
                    items_text = ", ".join(label[:-1]) + f", and {label[-1]}"
                    text = f"Deposit exceeded. Please return the {items_text} immediately"
            else:
                text = f"Deposit exceeded. Please return the {label} immediately"

            text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            filepath  = os.path.join(self.deposit_sounds_dir, f"deposit_{text_hash}.mp3")

            if not os.path.exists(filepath):
                gTTS(text=text, lang='en', slow=False).save(filepath)

            return filepath

        except Exception as e:
            print(f"Error generating deposit audio: {e}")
            return None

    def speak_deposit(self, label):
        try:
            if isinstance(label, str) and "," in label:
                label = [item.strip() for item in label.split(",")]

            filepath = self.generate_deposit_audio_file(label)

            if filepath and os.path.exists(filepath):
                self.play_mp3_async(filepath, volume=0.8)
            else:
                if isinstance(label, str):
                    self.speak_async(
                        f"Deposit exceeded. Please return the {label} immediately"
                    )
                elif isinstance(label, (list, tuple)):
                    if len(label) == 1:
                        self.speak_async(
                            f"Deposit exceeded. Please return the {label[0]} immediately"
                        )
                    elif len(label) == 2:
                        self.speak_async(
                            f"Deposit exceeded. Please return the "
                            f"{label[0]} and {label[1]} immediately"
                        )
                    else:
                        items_text = ", ".join(label[:-1]) + f", and {label[-1]}"
                        self.speak_async(
                            f"Deposit exceeded. Please return the {items_text} immediately"
                        )

        except Exception as e:
            print(f"Error in speak_deposit: {e}")
            try:
                self.speak_async("Deposit exceeded. Please return the items immediately")
            except Exception:
                pass

    def generate_door_audio_files(self):
        try:
            gTTS(text="Open the door",        lang='en', slow=False).save("sounds/door_open.mp3")
            gTTS(text="Door has been closed",  lang='en', slow=False).save("sounds/door_close.mp3")
            print("Door audio files generated")
        except Exception as e:
            print(f"Error generating door audio: {e}")

    def speak_door_open(self):
        self.play_mp3_sync("sounds/door_open.mp3", volume=0.8)

    def speak_door_close(self):
        self.play_mp3_sync("sounds/door_close.mp3", volume=0.8)

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
                except Exception as e:
                    print(f"gTTS error: {e}")
                    self.fallback_speak(text)

        threading.Thread(target=_speak, daemon=True).start()

    def speak_english(self, text):
        self.speak_async(text, lang='en')

    def speak_malay(self, text):
        self.speak_async(text, lang='ms')

    def fallback_speak(self, text):
        try:
            subprocess.run(['which', 'espeak'], check=True, capture_output=True)
            subprocess.run(
                ['espeak', '-s', '120', '-a', '200', '-p', '50', '-g', '3', text],
                check=False, capture_output=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.alternative_fallback(text)

    def alternative_fallback(self, text):
        try:
            subprocess.run(['which', 'festival'], check=True, capture_output=True)
            temp_file = '/tmp/tts_temp.txt'
            with open(temp_file, 'w') as f:
                f.write(text)
            subprocess.run(['festival', '--tts', temp_file], check=False, capture_output=True)
            os.remove(temp_file)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"No TTS fallback available. Would have spoken: {text}")

    def test_voice(self):
        self.speak_english("Testing voice clarity. Can you hear this clearly?")
        time.sleep(3)
        self.speak_malay("Ujian suara yang jelas. Boleh dengar dengan baik?")

    def cleanup(self):
        try:
            self.stop_all_audio()
            pygame.mixer.quit()
        except Exception:
            pass


# Global TTS manager instance
tts_manager = TTSManager()

# =====================================================================
# WEBSOCKET ENDPOINT
# =====================================================================

done = False


@app.websocket("/ws/track")
async def websocket_endpoint(websocket: WebSocket):
    global readyToProcess, done, current_pipeline_app

    deposit            = 0.0
    machine_id         = None
    machine_identifier = None
    user_id            = None
    transaction_id     = None
    websocket_sender   = None

    await websocket.accept()
    print("WebSocket connection accepted")

    tts_manager.speak_door_open()

    GPIO.output(DOOR_LOCK_PIN, GPIO.LOW)
    GPIO.output(LED_RED,       GPIO.LOW)
    GPIO.output(LED_GREEN,     GPIO.LOW)

    try:
        start_time     = time.time()
        readyToProcess = True

        while readyToProcess and time.time() - start_time < 5:
            door_sw = 1  # TODO: Replace with GPIO.input(DOOR_SWITCH_PIN)

            if door_sw == 1:
                await run_tracking(websocket)
                readyToProcess = False
            else:
                readyToProcess = True

        if not done:
            print("No tracking executed — sending initial update")
            callback = HailoDetectionCallback(
                websocket, deposit, machine_id,
                machine_identifier, user_id, transaction_id
            )
            if transaction_id:
                transaction_memory_manager.start_transaction(transaction_id)

            while not callback.tracking_data.shutdown_event.is_set():
                try:
                    current_data = callback.tracking_data.websocket_data_manager.get_current_data()
                    await websocket.send_json(current_data)
                    await asyncio.sleep(1)
                    break
                except Exception as e:
                    print(f"Error sending WebSocket data: {e}")
                    break

    except Exception as e:
        print(f"WebSocket tracking error: {e}")

    finally:
        print("WebSocket cleanup starting…")

        try:
            if 'callback' in locals() and hasattr(callback, 'transaction_id') \
               and callback.transaction_id:
                transaction_memory_manager.end_transaction(callback.transaction_id)
        except Exception as e:
            print(f"Error ending transaction: {e}")

        with pipeline_lock:
            if current_pipeline_app is not None:
                try:
                    current_pipeline_app.door_monitor_active = False
                    with current_pipeline_app.shutdown_lock:
                        if not current_pipeline_app.shutdown_called:
                            current_pipeline_app.shutdown()
                    time.sleep(2)
                except Exception as e:
                    print(f"Error stopping pipeline: {e}")
                finally:
                    current_pipeline_app = None

        await cleanup_websocket_sounds()

        GPIO.output(DOOR_LOCK_PIN, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(LED_GREEN,     GPIO.HIGH)
        GPIO.output(LED_RED,       GPIO.HIGH)

        if transaction_memory_manager.global_stats['total_transactions'] % 10 == 0:
            transaction_memory_manager.print_stats()

        await websocket.close()
        print("WebSocket connection closed")

# =====================================================================
# WEBSOCKET CLEANUP HELPER
# =====================================================================

async def cleanup_websocket_sounds():
    global camera_covered_sound_playing, price_alert_sound_playing

    camera_covered_sound_playing = False
    price_alert_sound_playing    = False
    tts_manager.stop_all_audio()
    print("WebSocket cleanup — all audio stopped")

# =====================================================================
# HEALTH CHECK AND MONITORING ENDPOINTS
# =====================================================================

@app.get("/health")
async def health_check():
    stats = transaction_memory_manager.get_stats()
    return {
        "status":              "healthy" if stats['current_memory_mb'] < 1000 else "warning",
        "memory_mb":           round(stats['current_memory_mb'], 2),
        "active_transactions": stats['active_transactions'],
        "total_transactions":  stats['total_transactions'],
        "uptime_hours":        round(
            (time.time() - app.start_time) / 3600, 2
        ) if hasattr(app, 'start_time') else 0
    }


@app.get("/stats")
async def get_stats():
    stats = transaction_memory_manager.get_stats()
    return {
        "memory": {
            "current_mb":   round(stats['current_memory_mb'], 2),
            "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
            "percent":      round(psutil.virtual_memory().percent, 2)
        },
        "transactions": {
            "active":   stats['active_transactions'],
            "total":    stats['total_transactions'],
            "cleanups": stats['total_cleanups']
        },
        "recent": stats['recent_transactions']
    }

# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def main():
    global mqtt_client

    parser = argparse.ArgumentParser(
        description='Smart Fridge Object Detection and Tracking System'
    )
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to listen on (default: 8000)')
    args = parser.parse_args()

    app.start_time = time.time()

    print("Creating required directories…")
    os.makedirs('saved_videos',    exist_ok=True)
    os.makedirs('camera_images',   exist_ok=True)
    os.makedirs('sounds',          exist_ok=True)
    os.makedirs('sounds/deposits', exist_ok=True)
    os.makedirs('sounds/closing',  exist_ok=True)
    print("Directories ready")

    print("Setting up audio alerts…")
    setup_cover_alert_sound()
    print("Camera cover alerts ready")

    generate_beep_file()
    print("Beep sound ready")

    setup_product_upload_alerts()
    print("Product upload alerts ready")

    if not os.path.exists("sounds/door_open.mp3") or \
       not os.path.exists("sounds/door_close.mp3"):
        tts_manager.generate_door_audio_files()
    print("Door audio ready")

    tts_manager.generate_common_deposit_messages()
    print("Deposit alerts ready")

    print("[Memory] Transaction memory management initialised")

    # ------------------------------------------------------------------
    # MQTT SETUP
    # Set machine ID in environment so mqtt_client topics resolve
    # correctly from the moment connect() is called.
    # ------------------------------------------------------------------
    os.environ['MACHINE_ID'] = str(MQTT_MACHINE_ID)
    mqtt_client = MQTTClient()
    mqtt_client.connect()
    print(f"[MQTT] Client created and connected for machine_id={MQTT_MACHINE_ID}")
    # ------------------------------------------------------------------

    atexit.register(GPIO.cleanup)
    atexit.register(lambda: mqtt_client.disconnect() if mqtt_client else None)
    print("GPIO and MQTT cleanup handlers registered")

    print("\n" + "="*60)
    print("SMART FRIDGE SYSTEM STARTED")
    print("="*60)
    print(f"Memory at startup: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
    print(f"Available memory:  {psutil.virtual_memory().available / 1024 / 1024:.1f}MB")
    print(f"WebSocket:         ws://{args.host}:{args.port}/ws/track")
    print(f"Health check:      http://{args.host}:{args.port}/health")
    print(f"Statistics:        http://{args.host}:{args.port}/stats")
    print(f"MQTT machine ID:   {MQTT_MACHINE_ID}")
    print(f"MQTT topics:       AIfridge/{MQTT_MACHINE_ID}/rpi/connectionStatus")
    print(f"                   AIfridge/{MQTT_MACHINE_ID}/rpi/doorStatus")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")

    uvicorn.run(
        "app_server:app",
        host=args.host,
        port=args.port,
        reload=False
    )


if __name__ == "__main__":
    main()

# =====================================================================
# END OF SMART FRIDGE DETECTION SYSTEM v2.1
# =====================================================================
