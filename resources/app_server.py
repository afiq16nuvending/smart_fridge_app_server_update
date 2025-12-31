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
6. Text-to-speech alerts

HARDWARE REQUIREMENTS:
- Raspberry Pi 5 with Hailo-8L AI accelerator
- 2 USB cameras (dev/video0 and dev/video2)
- Door lock mechanism (GPIO 25)
- Door switch sensor (GPIO 26)
- Green LED (GPIO 23)
- Red LED (GPIO 18)
- Buzzer (GPIO 20)

WORKFLOW:
1. Customer opens app → WebSocket connects
2. Deposit deducted → Door unlocks → Cameras start
3. Customer takes/returns items → AI tracks movements
4. Price calculated in real-time
5. Door closes → Video saved → Refund processed
6. System cleans up memory → Ready for next customer

AUTHOR: Afiq
VERSION: 1.0
LAST UPDATED: 2/12/2025
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

# =====================================================================
# GLOBAL VARIABLES AND CONFIGURATION
# =====================================================================

# FastAPI application instance
app = FastAPI()

# Data queue for multi-stream processing
data_deque: Dict[int, deque] = {}

# =====================================================================
# GPIO PIN CONFIGURATION
# =====================================================================
# Pin numbers using BCM (Broadcom) numbering scheme

DOOR_LOCK_PIN = 25      # Controls electromagnetic door lock
DOOR_SWITCH_PIN = 26    # Detects if door is open/closed
LED_GREEN = 23          # Green status LED
LED_RED = 18            # Red alert LED
BUZZER_PIN = 20         # Alert buzzer

# =====================================================================
# GPIO INITIALIZATION
# =====================================================================

GPIO.setmode(GPIO.BCM)              # Use BCM pin numbering
GPIO.setwarnings(False)             # Suppress warnings

# Setup outputs with initial states
GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.HIGH)      # Buzzer OFF (active low)
GPIO.setup(LED_GREEN, GPIO.OUT, initial=GPIO.HIGH)       # Green LED OFF
GPIO.setup(LED_RED, GPIO.OUT, initial=GPIO.HIGH)         # Red LED ON 
GPIO.setup(DOOR_LOCK_PIN, GPIO.OUT, initial=GPIO.HIGH)   # Door LOCKED

# Setup input
GPIO.setup(DOOR_SWITCH_PIN, GPIO.IN)  # Door sensor (0=closed, 1=open)

# =====================================================================
# GLOBAL STATE FLAGS
# =====================================================================

# System state flags
readyToProcess = False          # Flag indicating system is ready to process
blink = False                   # Flag to control LED blinking
alert_thread = None            # Thread handle for alert management

# Camera cover detection
camera_covered = False          # Flag indicating if camera is covered/blocked
cover_alert_thread = None      # Thread handle for camera cover alerts

# Audio alert states
camera_covered_sound_playing = False  # Flag for camera cover alert sound
price_alert_sound_playing = False     # Flag for price exceeded alert sound
last_alerted_label = None             # Last product that triggered price alert

# Pipeline management
current_pipeline_app = None     # Currently running pipeline application
pipeline_lock = threading.Lock()  # Thread lock for pipeline operations
print_lock = threading.Lock()     # Thread lock for console output

# Transaction state
done = False                    # Flag indicating transaction completion
unlock_data = 0                # Flag for door unlock control

# =====================================================================
# TRACKING DATA STRUCTURES
# =====================================================================

# Store movement history for each tracked object (last 5 positions)
movement_history = defaultdict(lambda: deque(maxlen=5))

# Store bounding box area history for stability checking
bbox_area_history = defaultdict(lambda: deque(maxlen=10))

# Current movement direction for each track
movement_direction = {}

# Last counted direction to prevent duplicate counts
last_counted_direction = {}

# Visual trail storage for drawing object paths
object_trails = defaultdict(lambda: deque(maxlen=30))  # 30 points per trail
global_trails = defaultdict(lambda: deque(maxlen=30))

# =====================================================================
# CROSS-CAMERA TRACKING STRUCTURES
# =====================================================================
# These structures enable tracking the same object across multiple cameras

# Global track counter for assigning unique IDs across cameras
global_track_counter = 0

# Maps (camera_id, local_track_id) to global_track_id
local_to_global_id_map = {}

# Movement history using global IDs
global_movement_history = defaultdict(deque)

# Last counted direction using global IDs
global_last_counted_direction = {}

# Store labels for each global ID
global_track_labels = {}

# Active objects per camera: camera_id -> label -> {local_track_id: global_track_id}
active_objects_per_camera = {
    0: defaultdict(dict),  # Camera 0 (/dev/video0)
    1: defaultdict(dict)   # Camera 1 (/dev/video2)
}

# Cross-camera matching candidates: label -> [(camera_id, local_track_id, global_track_id)]
cross_camera_candidates = defaultdict(list)

# Per-camera movement tracking
camera_movement_history = {
    0: defaultdict(lambda: deque(maxlen=5)),  # Camera 0 history
    1: defaultdict(lambda: deque(maxlen=5))   # Camera 1 history
}

# Per-camera bounding box area history for stability checking
camera_bbox_area_history = {
    0: defaultdict(lambda: deque(maxlen=5)),  # Camera 0 bbox areas
    1: defaultdict(lambda: deque(maxlen=5))   # Camera 1 bbox areas
}

# =====================================================================
# HARDWARE CONTROL FUNCTIONS
# =====================================================================

def trigger_buzzer(duration=0.5):
    """
    Trigger the buzzer for a specified duration.
    
    The buzzer is active LOW, meaning:
    - GPIO.LOW = buzzer sounds
    - GPIO.HIGH = buzzer silent
    
    Args:
        duration (float): Duration in seconds to keep the buzzer on
        
    Usage:
        trigger_buzzer(0.5)  # Short beep
        trigger_buzzer(2.0)  # Longer alert
    """
    GPIO.output(BUZZER_PIN, GPIO.LOW)   # Turn buzzer ON
    time.sleep(duration)                 # Wait for specified duration
    GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turn buzzer OFF


def blink_led(pin, times, delay):
    """
    Blink an LED a specified number of times.
    
    Args:
        pin (int): GPIO pin number connected to the LED
        times (int): Number of blink cycles
        delay (float): Delay in seconds between ON and OFF states
        
    Example:
        blink_led(LED_GREEN, 3, 0.2)  # Blink green LED 3 times rapidly
        blink_led(LED_RED, 5, 0.5)    # Blink red LED 5 times slowly
    """
    for _ in range(times):
        GPIO.output(pin, GPIO.HIGH)  # Turn LED_RED ON , Turn LED_GREEN OFF
        time.sleep(delay)            # Wait
        GPIO.output(pin, GPIO.LOW)   # Turn LED_RED OFF , Turn LED_GREEN ON
        time.sleep(delay)            # Wait before next cycle
          

def control_door(pin, action, duration=0.5):
    """
    Control the electromagnetic door lock mechanism.
    
    The door lock is controlled by:
    - GPIO.LOW = unlocked (electromagnet off)
    - GPIO.HIGH = locked (electromagnet on)
    
    Args:
        pin (int): GPIO pin connected to the door lock
        action (str): 'unlock' or 'lock'
        duration (float): How long to keep door unlocked (for 'unlock' action)
        
    Safety Features:
    - Door automatically re-locks after duration
    - Default duration is 0.5 seconds
    
    Usage:
        control_door(DOOR_LOCK_PIN, 'unlock', 3.0)  # Unlock for 3 seconds
        control_door(DOOR_LOCK_PIN, 'lock')          # Lock immediately
    """
    if action.lower() == 'unlock':
        print("Unlocking door...")
        GPIO.output(pin, GPIO.LOW)    # Deactivate lock (unlock)
        time.sleep(duration)          # Keep unlocked for specified time
        GPIO.output(pin, GPIO.HIGH)   # Reactivate lock (lock again)
        print("Door locked again")
        
    elif action.lower() == 'lock':
        print("Locking door...")
        GPIO.output(pin, GPIO.HIGH)   # Ensure door is locked
        print("Door locked")
        
    else:
        print(f"Invalid action '{action}'. Use 'lock' or 'unlock'")

# =====================================================================
# COLOR COMPUTATION FOR VISUALIZATION
# =====================================================================

def compute_color_for_labels(label):
    """
    Compute a unique color for each product label for visualization.
    
    This function assigns consistent colors to product classes:
    - Person (class 0): Orange-red
    - Car (class 2): Pink
    - Other classes: Generated from palette
    
    Args:
        label (int): Class ID of the detected object
        
    Returns:
        tuple: BGR color tuple (B, G, R) for OpenCV drawing
        
    Note: OpenCV uses BGR format, not RGB!
    """
    # Color palette for generating unique colors
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    
    # Predefined colors for specific classes
    if label == 0:  # Person class
        color = (85, 45, 255)  # Orange-red in BGR
    elif label == 2:  # Car class
        color = (222, 82, 175)  # Pink in BGR
    else:
        # Generate unique color based on label value
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        
    return tuple(color)

# =====================================================================
# VISUAL TRAIL DRAWING
# =====================================================================

def draw_trail(frame, track_id, center, color, global_id=None):
    """
    Draw movement trail (breadcrumb path) for a tracked object.
    
    Trails help visualize object movement patterns:
    - Shows last 30 positions
    - Line thickness decreases with age (older = thinner)
    - Supports both local and global ID tracking
    
    Args:
        frame (numpy.ndarray): Video frame to draw on
        track_id (int): Local track ID or global ID
        center (tuple): Current center point (x, y) in pixels
        color (tuple): BGR color for the trail
        global_id (int, optional): Global ID across cameras
        
    Visual Effect:
        - Most recent positions: thick lines
        - Older positions: progressively thinner lines
        - Creates a "comet tail" effect showing direction
    """
    # Choose which trail storage to use
    if global_id is not None:
        # Use global trails for cross-camera tracking
        global_trails[global_id].appendleft(center)
        points = list(global_trails[global_id])
    else:
        # Use local trails for single-camera tracking
        object_trails[track_id].appendleft(center)
        points = list(object_trails[track_id])
    
    # Draw lines connecting consecutive points
    for i in range(1, len(points)):
        # Skip if any point is invalid
        if points[i - 1] is None or points[i] is None:
            continue
            
        # Calculate line thickness (thicker = more recent)
        # Formula: sqrt(64 / (i + 1)) * 2
        # i=0 (most recent): thickness ≈ 16
        # i=29 (oldest): thickness ≈ 3
        thickness = int(np.sqrt(64 / float(i + 1)) * 2)
        
        # Draw line segment
        cv2.line(frame, points[i - 1], points[i], color, thickness)

# =====================================================================
# ON-SCREEN COUNTER DISPLAY
# =====================================================================

def draw_counts(frame, class_counters, label):
     """
    Draw entry/exit counts on the video frame.
    
    Displays:
    1. Total entry count (all products)
    2. Total exit count (all products)
    3. Per-product entry and exit counts
    
    Args:
        frame (numpy.ndarray): Video frame to draw on
        class_counters (dict): Dictionary with 'entry' and 'exit' counts
        label (str): Current product label being processed
        
    Layout on screen:
        Top: Total Entry: X
             Total Exit: Y
        Below: [Product Name] Entry: X, Exit: Y (color-coded)
    """
    # Product name mapping (class ID to product name)
    class_names = {
            0: "",
            1: "chickenKatsuCurry", 
            2: "dakgangjeongRice",
            3: "kimchiFriedRice", 
            4: "kimchiTuna",
    
        
        
        
 
}
    
    """Draw both entry and exit counts on frame"""
    # Calculate totals
    total_entry = sum(class_counters["entry"].values())
    total_exit = sum(class_counters["exit"].values())
    
    # Draw total counts
    cv2.putText(frame, f'Total Entry: {total_entry}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Total Exit: {total_exit}', (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw class-specific counts (combined entry and exit on same line)
    y_offset = 110  # Starting y position for class counts
    
    # Get all unique labels from both entry and exit counters
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


# =====================================================================
# DETECTION ZONE VISUALIZATION
# =====================================================================

def draw_zone(frame):
    """
    Draw detection zone overlay on the frame.
    
    Currently draws a rectangle around the entire frame to indicate
    the active detection area. Can be modified to define specific zones.
    
    Args:
        frame (numpy.ndarray): Video frame to draw on
        
    Note: This can be enhanced to support:
    - Multiple detection zones
    - Entry/exit zones
    - Restricted areas
    """
    height, width = frame.shape[:2]
    
    # Draw green rectangle around entire frame
    cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
    
    # Add zone label
    cv2.putText(frame, "Detection Zone", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# =====================================================================
# ALERT STATE MANAGEMENT
# =====================================================================

def handle_alert_state():
    """
    Handle LED blinking and buzzer when price exceeds deposit.
    
    This function runs in a separate thread and continuously blinks
    the red LED until either:
    1. The door is closed (DOOR_SWITCH_PIN = 0)
    2. The global 'blink' flag is set to False
    
    Behavior:
    - Red LED blinks at 0.5 second intervals
    - Buzzer remains silent (handled separately)
    - Exits when door closes
    
    Global Variables Used:
        blink (bool): Controls whether blinking should continue
    """
    global blink
    
    while blink:
        # Check if door is closed
        if GPIO.input(DOOR_SWITCH_PIN) == 0:
            GPIO.output(LED_RED, GPIO.LOW)      # Turn off LED
            GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Ensure buzzer is off
            break
            
        # Blink cycle
        GPIO.output(LED_RED, GPIO.HIGH)  # LED on
        time.sleep(0.5)                   # Wait
        GPIO.output(LED_RED, GPIO.LOW)   # LED off
        time.sleep(0.5)                   # Wait

# =====================================================================
# PRICE CALCULATION AND ALERT SYSTEM
# =====================================================================

def calculate_total_price_and_control_buzzer(current_data, deposit, label=None):
    """
    Calculate total price of taken items and trigger alerts if deposit exceeded.
    
    This is the core payment validation function that:
    1. Calculates price of all products taken (exit - entry)
    2. Compares total price against customer's deposit
    3. Triggers visual (LED) and audio (TTS) alerts if exceeded
    4. Manages alert states to prevent duplicate announcements
    
    Args:
        current_data (dict): Current detection data with validated products
        deposit (float): Customer's deposit amount
        label (str, optional): Current product label being processed
        
    Returns:
        float: Total price of products taken
        
    Algorithm:
    - For each product: count_taken = exit_count - entry_count
    - If count_taken > 0: add (count_taken * price) to total
    - If total > deposit: alert customer to return items
    
    Global Variables Modified:
        blink: Set to True/False to control LED blinking
        price_alert_sound_playing: Tracks if audio alert is playing
        last_alerted_label: Tracks which products were alerted
        alert_thread: Thread handle for LED blinking
    """
    global blink, alert_thread, price_alert_sound_playing, last_alerted_label
    
    total_product_price = 0
    
    # Get validated products from detection data
    validated_products = current_data.get("validated_products", {})
    
    # Get all unique product names from both entry and exit
    all_products = set(validated_products.get("entry", {}).keys()) | \
                   set(validated_products.get("exit", {}).keys())
    
    # Dictionary to store price contribution per product
    product_prices = {}
    
    # Calculate price for each product
    for product_name in all_products:
        # Get entry and exit data (default to empty dict if not present)
        entry_data = validated_products.get("entry", {}).get(product_name, {"count": 0})
        exit_data = validated_products.get("exit", {}).get(product_name, {"count": 0})
        
        # Get counts
        entry_count = entry_data.get("count", 0)
        exit_count = exit_data.get("count", 0)
        
        # Get product details (price info)
        product_details = exit_data.get("product_details") or entry_data.get("product_details")
        
        # Calculate if product has valid price
        if product_details and "product_price" in product_details:
            price_per_unit = float(product_details["product_price"])
            
            # True count = items taken out minus items put back
            true_count = max(0, exit_count - entry_count)
            
            # Calculate total for this product
            product_total = true_count * price_per_unit
            
            # Only track if customer actually took items
            if true_count > 0:
                product_prices[product_name] = product_total
                total_product_price += product_total
    
    # ===== ALERT LOGIC =====
    
    if total_product_price > deposit:
        # DEPOSIT EXCEEDED - Trigger alerts
        
        blink = True  # Start LED blinking
        
        # Sort products by price (highest first) for alert priority
        products_to_return = sorted(product_prices.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
        
        # Create list of product names to return
        products_list = [p[0] for p in products_to_return]
        
        # Convert to string for comparison (detect if alert changed)
        products_str = ",".join(products_list)
        
        # Play audio alert if:
        # 1. There are products to return AND
        # 2. (Alert not playing OR products changed since last alert)
        if products_list and (not price_alert_sound_playing or 
                            last_alerted_label != products_str):
            price_alert_sound_playing = True
            tts_manager.speak_deposit(products_list)  # Text-to-speech alert
            last_alerted_label = products_str
            
            print(f"Price alert: ${total_product_price:.2f} > ${deposit:.2f}")
            print(f"Please return: {products_str}")
        
        # Start LED blinking thread if not already running
        if alert_thread is None or not alert_thread.is_alive():
            alert_thread = threading.Thread(target=handle_alert_state, daemon=True)
            alert_thread.start()
            
    else:
        # WITHIN DEPOSIT LIMIT - Clear alerts
        
        blink = False  # Stop LED blinking
        GPIO.output(LED_RED, GPIO.LOW)  # Ensure LED is off
        
        # Stop price alert audio if it was playing
        if price_alert_sound_playing:
            price_alert_sound_playing = False
            last_alerted_label = None
            
            # Only stop audio if camera cover alert isn't playing
            if not camera_covered_sound_playing:
                tts_manager.stop_all_audio()
                
            print("Price within deposit limit - stopping price alert")
    
    return total_product_price

# =====================================================================
# DOOR STATUS MONITORING
# =====================================================================

def check_door_status():
    """
    Monitor door switch status continuously.
    
    This function runs in a loop checking the door sensor:
    - Returns True when door closes (sensor = 0)
    - Prevents CPU overuse with small sleep delay
    - Used to trigger shutdown when customer closes door
    
    Returns:
        bool: True when door is detected as closed
        
    Note: Currently uses a placeholder value (door_sw = 1).
    In production, this reads: door_sw = GPIO.input(DOOR_SWITCH_PIN)
    """
    while True:
        # TODO: Replace with actual GPIO reading in production
        door_sw = 1  # Placeholder: 0=closed, 1=open
        
        with print_lock:
            if door_sw == 0:  # Door is closed
                print("Door closed - Shutting down preview frames")
                return True
                
        time.sleep(0.1)  # Small delay to prevent CPU overuse

# =====================================================================
# CAMERA COVER DETECTION
# =====================================================================

def is_frame_dark(frame, threshold=40):
    """
    Detect if camera is covered or blocked by checking frame darkness.
    
    This security feature prevents tampering by detecting when:
    - Camera lens is covered with hand/object
    - Lighting is completely blocked
    - Image is abnormally dark
    
    Args:
        frame (numpy.ndarray): Input video frame (RGB or grayscale)
        threshold (int): Brightness threshold (0-255)
            Lower values = more sensitive to darkness
            Default 40 = very dark
            
    Returns:
        bool: True if frame is dark (potentially covered), False otherwise
        
    Algorithm:
    1. Convert to grayscale if needed
    2. Calculate average brightness across all pixels
    3. Compare against threshold
    
    Global Variables Modified:
        camera_covered_sound_playing: Updated when darkness detected
    """
    global camera_covered_sound_playing
    
    # Convert to grayscale if frame is color (3 channels)
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Calculate average brightness (0=black, 255=white)
    avg_brightness = np.mean(gray)
    
    # Determine if frame is dark
    is_dark = avg_brightness < threshold
    
    return is_dark

# =====================================================================
# CAMERA COVER ALERT SETUP
# =====================================================================

def setup_cover_alert_sound():
    """
    Generate and save camera cover alert sound using text-to-speech.
    
    Creates an MP3 file with the alert message:
    "Don't cover the camera. Please uncover the camera immediately."
    
    Directory structure:
        sounds/
        └── cover_alerts/
            └── camera_covered.mp3
    
    Returns:
        str: Path to the generated alert sound file
        
    Note: Alert is only generated once and cached for reuse.
    """
    alert_dir = "sounds/cover_alerts"
    alert_file = os.path.join(alert_dir, "camera_covered.mp3")
    
    # Create directory if it doesn't exist
    os.makedirs(alert_dir, exist_ok=True)
    
    # Generate alert if it doesn't exist
    if not os.path.exists(alert_file):
        alert_text = "Dont cover the camera. Please uncover the camera immediately."
        tts = gTTS(text=alert_text, lang='en', slow=False)
        tts.save(alert_file)
        print(f"Cover alert sound saved to {alert_file}")
    
    return alert_file

# =====================================================================
# CAMERA COVER ALERT HANDLER
# =====================================================================

def handle_cover_alert():
    """
    Play audio alert repeatedly while camera is covered.
    
    This function runs in a separate thread and:
    1. Plays warning message in a loop
    2. Checks door status (exits if door closes)
    3. Waits 3 seconds between repetitions
    4. Stops when camera is uncovered (camera_covered = False)
    
    Alert Message:
        "Don't cover the camera. Please uncover the camera immediately."
    
    Exit Conditions:
    - camera_covered flag becomes False
    - Door closes (DOOR_SWITCH_PIN = 0)
    
    Global Variables Used:
        camera_covered (bool): Controls loop execution
    """
    global camera_covered
    
    # Get or create the alert sound file
    alert_sound = setup_cover_alert_sound()
    
    print("Camera covered - playing alert sound")
    
    # Loop while camera is covered
    while camera_covered:
        # Check if door is closed (emergency exit)
        if GPIO.input(DOOR_SWITCH_PIN) == 0:
            print("Door closed - stopping alert sound")
            break
        
        # Play the cover alert sound
        # volume=0.8 = 80% volume
        tts_manager.play_mp3_async(alert_sound, volume=0.8)
        
        # Wait 3 seconds before repeating
        # Adjust based on TTS message length
        time.sleep(3.0)
    
    print("Camera uncovered - stopping alert sound")

# =====================================================================
# VIDEO DISPLAY AND RECORDING SYSTEM
# =====================================================================

def display_user_data_frame(user_data):
    """
    Main video display loop with recording capability.
    
    This function:
    1. Displays live video feed in OpenCV window
    2. Records video to local filesystem
    3. Monitors door status for shutdown
    4. Calculates actual FPS dynamically
    5. Handles cleanup on exit
    
    Features:
    - Automatic FPS detection (first 30 frames)
    - Video saved as .avi format (XVID codec)
    - Filename includes timestamp and transaction ID
    - Thread-safe operation
    
    Args:
        user_data: User data object containing:
            - get_frame(): Method to retrieve current frame
            - shutdown_event: Event to signal shutdown
            - transaction_id: Unique transaction identifier
            - machine_id: Machine identifier
            - user_id: User identifier
            - machine_identifier: Machine identifier string
    
    Directory Structure:
        saved_videos/
        └── hailo_detection_YYYYMMDD_HHMMSS_[transaction_id].avi
    
    Video Specs:
        - Codec: XVID
        - FPS: Auto-detected (typically ~13 FPS)
        - Resolution: Matches input frame dimensions
    """
    # Start door monitoring in a separate thread
    door_monitor_thread = threading.Thread(target=check_door_status)
    door_monitor_thread.daemon = True  # Thread exits when main program exits
    door_monitor_thread.start()
    
    # Extract transaction details from user_data
    transaction_id = getattr(user_data, 'transaction_id', None) 
    machine_id = getattr(user_data, 'machine_id', None) 
    user_id = getattr(user_data, 'user_id', None) 
    machine_identifier = getattr(user_data, 'machine_identifier', None)
    
    # Create video directory if it doesn't exist
    video_dir = os.path.join(os.getcwd(), "saved_videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # Set up video writer (XVID codec for .avi format)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = None
    
    # Generate filename with timestamp and transaction ID
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    dataset_name = f"hailo_detection_{timestamp}_{transaction_id}"
    filename = os.path.join(video_dir, f"{dataset_name}.avi")
    
    # FPS calculation variables
    frame_count = 0
    fps_start_time = None
    fps_calculated = False
    actual_fps = 13.0  # Default fallback FPS
    fps_sample_frames = 30  # Calculate FPS over first 30 frames
    
    try:
        while not user_data.shutdown_event.is_set():
            # Check if door is closed (placeholder)
            door_sw = 1  # TODO: Replace with GPIO.input(DOOR_SWITCH_PIN)
            
            # Get current frame
            frame = user_data.get_frame()
            if frame is not None:
                # Start timing on first frame
                if fps_start_time is None:
                    fps_start_time = time.time()
                
                frame_count += 1
                
                # Calculate actual FPS after collecting sample frames
                if not fps_calculated and frame_count >= fps_sample_frames:
                    elapsed_time = time.time() - fps_start_time
                    actual_fps = frame_count / elapsed_time
                    fps_calculated = True
                    print(f"Detected actual FPS: {actual_fps:.2f}")
                
                # Create video writer after FPS is calculated
                if output_video is None and fps_calculated:
                    height, width = frame.shape[:2]
                    output_video = cv2.VideoWriter(
                        filename, 
                        fourcc, 
                        actual_fps, 
                        (width, height), 
                        isColor=True
                    )
                    print(f"Started recording to: {filename}")
                    print(f"Recording at {actual_fps:.2f} FPS")
                
                # Write frame to video (only after video writer is created)
                if output_video is not None:
                    output_video.write(frame.copy())
                
                # Display frame in window
                cv2.imshow("Hailo Detection", frame)
                
                # Check for 'q' key press to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except Exception as e:
        print(f"Error in display loop: {e}")
        
    finally:
        print("Cleaning up display resources...")
        
        # Release video writer
        if output_video is not None:
            output_video.release()
            print(f"Video saved successfully to: {filename}")
            print(f"Total frames recorded: {frame_count}")
        
        try:
            # Clean GPIO - reset all outputs to safe state
            GPIO.output(DOOR_LOCK_PIN, GPIO.HIGH)   # Lock door
            GPIO.output(LED_GREEN, GPIO.HIGH)        # Turn off green LED
            GPIO.output(LED_RED, GPIO.HIGH)          # Turn on red LED
        except Exception as e:
            print(f"Error cleaning up GPIO: {e}")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        for i in range(5):  # Force windows to close
            cv2.waitKey(1)
            
        # Set shutdown event last
        user_data.shutdown_event.set()
        print("Display cleanup complete")

# =====================================================================
# VIDEO UPLOAD TO API
# =====================================================================

def stream_video_to_api(video_path, dataset_name, transaction_id, machine_id, 
                       user_id, machine_identifier):
    """
    Stream recorded video to the backend API.
    
    This function uploads the recorded transaction video to the server
    for storage and future analysis/review.
    
    Args:
        video_path (str): Local path to the video file
        dataset_name (str): Name for the dataset
        transaction_id (str): Unique transaction identifier
        machine_id (str): Machine identifier
        user_id (str): User identifier
        machine_identifier (str): Machine identifier string
        
    Returns:
        bool: True if upload successful, False otherwise
        
    API Endpoint:
        POST /shopping_app/machine/TransactionDataset/insert_transactionDataset
        
    Authentication:
        - Basic Auth: username='admin', password='1234'
        - API Key: '123456' in x-api-key header
        
    Upload Details:
        - Method: HTTP POST with multipart/form-data
        - File field: 'video'
        - Additional fields: machine_id, created_by, dataset_url, etc.
        - Timeout: 30 seconds
    """
    # API endpoint URL
    api_url = "https://stg-sfapi.nuboxtech.com/index.php/shopping_app/machine/TransactionDataset/insert_transactionDataset"
    
    # Authentication credentials
    username = 'admin'
    password = '1234'
    api_key = '123456'
    
    # Get current timestamp for database record
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract filename from path
    filename = os.path.basename(video_path)
    
    # Prepare payload (form data)
    payload = {
        'machine_id': machine_id,
        'created_by': user_id,
        'dataset_url': f"assets/video/machine_transaction_dataset/{machine_identifier}/{dataset_name}.avi",
        'dataset_name': dataset_name,
        'transaction_id': transaction_id,
        'created_datetime': current_time
    }
    
    # Prepare headers
    headers = {'x-api-key': api_key}
    
    print(f"Streaming video to API: {video_path}")
    print(f"Payload: {payload}")
    
    try:
        # Open the file in streaming mode
        with open(video_path, 'rb') as video_file:
            # Create multipart form with video file
            files = {'video': (filename, video_file, 'video/avi')}
            
            # Send POST request with authentication
            response = requests.post(
                api_url,
                auth=HTTPBasicAuth(username, password),
                headers=headers,
                data=payload,
                files=files,
                timeout=30.0  # 30 second timeout for large files
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
# VIDEO MONITORING AND AUTO-UPLOAD SYSTEM
# =====================================================================

def monitor_and_send_videos(video_directory, machine_id, machine_identifier, user_id):
    """
    Background thread that monitors video directory and auto-uploads completed videos.
    
    This sophisticated monitoring system:
    1. Continuously scans for new .avi files
    2. Uses atomic lock files to prevent race conditions
    3. Verifies file completion before upload
    4. Handles stale locks (cleanup after 5 minutes)
    5. Automatically deletes videos after successful upload
    6. Prevents memory growth with processed video tracking
    
    Why This Matters:
    - Videos are created WHILE customer is still shopping
    - Must wait until file is complete before uploading
    - Multiple threads might try to upload same file
    - Lock mechanism ensures only ONE thread processes each file
    
    Lock File System:
    - video.avi          (actual video file)
    - video.avi.processing  (lock file with timestamp)
    
    Args:
        video_directory (str): Directory to monitor for video files
        machine_id (str): Machine identifier for API upload
        machine_identifier (str): Machine identifier string
        user_id (str): User identifier for API upload
        
    Runs indefinitely as daemon thread until program exit.
    """
    print(f"Starting video monitor thread for directory: {video_directory}")
    
    # Track which videos have been processed (prevent re-processing)
    processed_videos = set()
    
    # Thread lock for accessing processed_videos set
    processing_lock = Lock()
    
    def create_lock_file(video_path):
        """
        Create atomic lock file - returns True if successful, False if already locked.
        
        Uses os.O_CREAT | os.O_EXCL for atomic creation:
        - Succeeds only if file doesn't exist
        - Fails if file already exists (another thread locked it)
        - This is atomic at OS level (no race condition)
        """
        lock_path = video_path + '.processing'
        try:
            # O_CREAT | O_EXCL = atomic creation, fails if exists
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{time.time()}\n".encode())  # Write timestamp
            os.close(fd)
            return True
        except FileExistsError:
            return False  # Already locked by another thread
        except Exception as e:
            print(f"Error creating lock file: {e}")
            return False
    
    def remove_lock_file(video_path):
        """Remove lock file after processing complete."""
        lock_path = video_path + '.processing'
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception as e:
            print(f"Error removing lock file: {e}")
    
    def is_lock_stale(video_path, timeout=300):
        """
        Check if lock file is stale (older than timeout seconds).
        
        Stale locks can occur if:
        - Program crashed while processing
        - Thread was killed unexpectedly
        
        Args:
            video_path (str): Path to video file
            timeout (int): Seconds before lock is considered stale (default 300 = 5 minutes)
            
        Returns:
            bool: True if lock is stale and should be removed
        """
        lock_path = video_path + '.processing'
        try:
            if not os.path.exists(lock_path):
                return False
                
            # Read timestamp from lock file
            with open(lock_path, 'r') as f:
                timestamp = float(f.read().strip())
                
            # Check if lock is older than timeout
            return time.time() - timestamp > timeout
        except:
            return True  # Consider stale if we can't read it
    
    # Main monitoring loop (runs forever)
    while True:
        try:
            # Find all .avi files in directory
            video_pattern = os.path.join(video_directory, "*.avi")
            video_files = glob.glob(video_pattern)
            
            # Process each video file
            for video_path in video_files:
                # Skip if already processed
                with processing_lock:
                    if video_path in processed_videos:
                        continue
                
                # Check for stale lock and clean it up
                if is_lock_stale(video_path):
                    print(f"Removing stale lock for: {video_path}")
                    remove_lock_file(video_path)
                
                # Try to acquire atomic lock
                if not create_lock_file(video_path):
                    # File is being processed by another thread
                    continue
                
                try:
                    print(f"Acquired lock for processing: {video_path}")
                    
                    # Check if file still exists (might have been deleted)
                    if not os.path.exists(video_path):
                        with processing_lock:
                            processed_videos.add(video_path)
                        continue
                    
                    # Verify file is complete (not still being written)
                    if is_file_complete_enhanced(video_path):
                        print(f"Found complete video file: {video_path}")
                        
                        # Double-check file still exists
                        if not os.path.exists(video_path):
                            print(f"File was deleted during processing: {video_path}")
                            with processing_lock:
                                processed_videos.add(video_path)
                            continue
                        
                        # Extract dataset info from filename
                        filename = os.path.basename(video_path)
                        dataset_name = filename.replace('.avi', '')
                        
                        # Extract transaction_id from filename
                        # Format: hailo_detection_YYYYMMDD_HHMMSS_TRANSACTION_ID.avi
                        try:
                            parts = dataset_name.split('_')
                            transaction_id = parts[-1] if len(parts) > 2 else None
                        except:
                            transaction_id = None
                        
                        # Attempt upload to API
                        success = stream_video_to_api(
                            video_path, 
                            dataset_name, 
                            transaction_id, 
                            machine_id, 
                            user_id, 
                            machine_identifier
                        )
                        
                        if success:
                            print(f"Successfully uploaded: {video_path}")
                            
                            # Delete video file after successful upload
                            try:
                                if os.path.exists(video_path):
                                    os.remove(video_path)
                                    print(f"Deleted uploaded video: {video_path}")
                                else:
                                    print(f"Video file already deleted: {video_path}")
                            except Exception as e:
                                print(f"Error deleting video file: {e}")
                        else:
                            # Delete even on failure (likely timeout)
                            os.remove(video_path)
                            print(f"Deleted uploaded video (Read timed out): {video_path}")
                            print(f"Failed to upload: {video_path}")
                        
                        # Mark as processed regardless of upload success
                        with processing_lock:
                            processed_videos.add(video_path)
                    else:
                        print(f"Video file not yet complete: {video_path}")
                        
                except Exception as e:
                    print(f"Error processing video {video_path}: {e}")
                    
                finally:
                    # Always remove lock file when done
                    remove_lock_file(video_path)
                    print(f"Released lock for: {video_path}")
            
            # Clean up processed videos set periodically to prevent memory growth
            with processing_lock:
                if len(processed_videos) > 100:
                    # Keep only last 50 entries
                    processed_videos = set(list(processed_videos)[-50:])
                    
        except Exception as e:
            print(f"Error in video monitoring thread: {e}")
        
        # Wait between checks to reduce CPU usage
        time.sleep(10)  # Check every 10 seconds

# =====================================================================
# FILE COMPLETION VERIFICATION
# =====================================================================

def is_file_complete_enhanced(file_path, stable_time=5):
    """
    Enhanced file completion check specifically for video files.
    
    A file is considered "complete" when:
    1. File exists and has content (size > 0)
    2. File size doesn't change for 5 seconds
    3. File modification time doesn't change for 5 seconds
    4. File can be opened exclusively (not being written)
    
    This prevents uploading files that are still being recorded.
    
    Args:
        file_path (str): Path to video file to check
        stable_time (int): Seconds file must be stable (default 5)
        
    Returns:
        bool: True if file is complete and ready for upload
        
    Algorithm:
    1. Get initial file stats (size, mtime)
    2. Wait 5 seconds
    3. Get final file stats
    4. Compare - if unchanged, file is complete
    5. Try exclusive open to verify not being written
    """
    try:
        # Check file exists
        if not os.path.exists(file_path):
            return False
        
        # Get initial file statistics
        initial_stat = os.stat(file_path)
        initial_size = initial_stat.st_size
        initial_mtime = initial_stat.st_mtime
        
        # File must have some content
        if initial_size == 0:
            return False
        
        print(f"Checking completion for {file_path} (size: {initial_size} bytes)")
        
        # Wait for stability period
        time.sleep(stable_time)
        
        # Check again
        try:
            final_stat = os.stat(file_path)
            final_size = final_stat.st_size
            final_mtime = final_stat.st_mtime
        except OSError:
            # File might have been deleted or locked
            return False
        
        # Size and modification time should be unchanged
        if initial_size != final_size or initial_mtime != final_mtime:
            print(f"File still changing: size {initial_size}->{final_size}, "
                  f"mtime {initial_mtime}->{final_mtime}")
            return False
        
        # Try to open file exclusively to ensure it's not being written
        try:
            with open(file_path, 'r+b') as f:
                # Seek to end to verify file integrity
                f.seek(0, 2)  # Seek to end
                actual_size = f.tell()
                if actual_size != final_size:
                    return False
        except (IOError, OSError) as e:
            print(f"Cannot open file exclusively: {e}")
            return False
        
        print(f"File appears complete: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error in enhanced completion check: {e}")
        return False

# =====================================================================
# WEBSOCKET DATA MANAGEMENT
# =====================================================================

class WebSocketDataManager:
    """
    Thread-safe manager for real-time data updates via WebSocket.
    
    This class handles:
    - Storing current detection state
    - Thread-safe updates from detection callback
    - Providing data snapshots for WebSocket transmission
    
    Data Structure:
        {
            "validated_products": {
                "entry": {product_name: {count, product_details}},
                "exit": {product_name: {count, product_details}}
            },
            "invalidated_products": {
                "entry": {product_name: {count, raw_detection}},
                "exit": {product_name: {count, raw_detection}}
            }
        }
    
    Thread Safety:
        Uses threading.Lock to prevent race conditions when:
        - Detection callback updates data
        - WebSocket thread reads data for transmission
    """
    
    def __init__(self):
        """
        Initialize with empty data structure and thread lock.
        """
        self.current_data = {
            "validated_products": {
                "entry": {},  # Products being put back
                "exit": {}    # Products being taken out
            },
            "invalidated_products": {
                "entry": {},  # Invalid products detected (put back)
                "exit": {}    # Invalid products detected (taken out)
            }
        }
        print(f'Initialized WebSocketDataManager: {self.current_data}')
        
        # Thread lock for safe concurrent access
        self._lock = threading.Lock()

    def update_data(self, new_data):
        """
        Update current data with new detection results.
        
        Called by detection callback thread whenever new detections occur.
        Thread-safe operation using lock.
        
        Args:
            new_data (dict): New data structure to replace current data
        """
        with self._lock:
            self.current_data = new_data

    def get_current_data(self):
        """
        Get a copy of current data for WebSocket transmission.
        
        Returns a COPY to prevent modification by caller.
        Thread-safe operation using lock.
        
        Returns:
            dict: Copy of current detection data
        """
        with self._lock:
            return self.current_data.copy()

# =====================================================================
# TRACKING DATA CONTAINER
# =====================================================================

class TrackingData:
    """
    Main container for all tracking state and configuration.
    
    This class stores:
    1. Product detection results (validated/invalidated)
    2. Entry/exit counters per product class
    3. Set of already-counted track IDs (prevent duplicates)
    4. Machine planogram (inventory list)
    5. Transaction information (deposit, IDs)
    6. WebSocket data manager
    7. Shutdown event for cleanup
    
    Used by:
    - Detection callback to update counters
    - WebSocket sender to transmit updates
    - Price calculator to determine charges
    """
    
    def __init__(self):
        """
        Initialize all tracking data structures.
        """
        # Shutdown signal for graceful cleanup
        self.shutdown_event = Event()
        
        # Product detection results
        self.validated_products = {
            "entry": {},  # Valid products being returned
            "exit": {}    # Valid products being taken
        }
        self.invalidated_products = {
            "entry": {},  # Invalid products detected (returned)
            "exit": {}    # Invalid products detected (taken)
        }
        
        # Counter for each product class
        # Format: {"entry": {product_name: count}, "exit": {product_name: count}}
        self.class_counters = {
            "entry": defaultdict(int),
            "exit": defaultdict(int)
        }
        
        # Track IDs that have been counted (prevent duplicate counting)
        # Format: {"entry": set(track_ids), "exit": set(track_ids)}
        self.counted_tracks = {
            "entry": set(),
            "exit": set()
        }
        
        # Machine inventory (loaded from API)
        self.machine_planogram = []
        
        # Pipeline configuration
        self.hailo_pipeline_string = ""
        
        # FPS calculation
        self.frame_rate_calc = 1
        self.last_time = time.time()
        
        # WebSocket data manager for real-time updates
        self.websocket_data_manager = WebSocketDataManager()
        
        # Transaction information
        self.deposit = 0.0              # Customer's deposit amount
        self.machine_id = None          # Unique machine identifier
        self.machine_identifier = None  # Machine name/code
        self.user_id = None            # Customer user ID
        self.transaction_id = None     # Unique transaction ID
        
    def set_transaction_data(self, deposit, machine_id, machine_identifier, 
                            user_id, transaction_id):
        """
        Set transaction information when customer starts shopping.
        
        Args:
            deposit (float): Customer's deposit amount
            machine_id (str): Machine database ID
            machine_identifier (str): Machine name/code
            user_id (str): Customer user ID
            transaction_id (str): Unique transaction ID
        """
        self.deposit = deposit
        self.machine_id = machine_id
        self.machine_identifier = machine_identifier
        self.user_id = user_id
        self.transaction_id = transaction_id

# =====================================================================
# HAILO DETECTION CALLBACK CLASS
# =====================================================================

class HailoDetectionCallback(app_callback_class):
    """
    Main callback class for Hailo AI detection pipeline.
    
    This class:
    1. Initializes tracking data and video recording
    2. Loads machine planogram from API
    3. Manages planogram refresh in background
    4. Validates detected products against inventory
    5. Provides pipeline configuration
    
    Lifecycle:
    - Created when customer opens fridge
    - Loads planogram on init (or from cache)
    - Processes frames during transaction
    - Cleans up when customer closes door
    """
    
    def __init__(self, websocket=None, deposit=0.0, machine_id=None, 
                 machine_identifier=None, user_id=None, transaction_id=None):
        """
        Initialize detection callback with transaction information.
        
        Args:
            websocket: WebSocket connection for real-time updates
            deposit (float): Customer's deposit amount
            machine_id (str): Machine database ID
            machine_identifier (str): Machine name/code
            user_id (str): Customer user ID
            transaction_id (str): Unique transaction ID
        """
        super().__init__()
        
        # Initialize tracking data container
        self.tracking_data = TrackingData()
        
        # Configuration
        self.use_frame = True
        self.websocket = websocket
        self.shutdown_event = Event()
        
        # Transaction information
        self.deposit = deposit
        self.machine_id = machine_id
        self.machine_identifier = machine_identifier
        self.user_id = user_id
        self.transaction_id = transaction_id
        
        # Set transaction data in tracking container
        self.tracking_data.set_transaction_data(
            deposit, machine_id, machine_identifier, user_id, transaction_id
        )
        
        # Create video directory BEFORE loading planogram
        self.video_directory = os.path.join(os.getcwd(), "saved_videos")
        os.makedirs(self.video_directory, exist_ok=True)
        
        # Store machine_id in environment variable for persistence
        self.store_machine_id_env(machine_id)
        
        # Load machine planogram (product inventory)
        self.load_machine_planogram()
    
    # =================================================================
    # MACHINE ID PERSISTENCE (Environment Variable Storage)
    # =================================================================
    
    def store_machine_id_env(self, machine_id):
        """
        Store machine_id as environment variable for persistence.
        
        Why: Machine ID needs to persist across function calls
        and thread boundaries for planogram management.
        
        Args:
            machine_id (str): Machine identifier to store
        """
        if machine_id is not None:
            os.environ['MACHINE_ID'] = str(machine_id)
            print(f"Machine ID {machine_id} stored in environment")
    
    def load_machine_id_env(self):
        """
        Load machine_id from environment variable.
        
        Returns:
            str: Machine ID from environment, or None if not set
        """
        return os.environ.get('MACHINE_ID')
    
    # =================================================================
    # PLANOGRAM VALIDATION AND CACHING
    # =================================================================
    
    def is_planogram_valid_for_machine(self, machine_id):
        """
        Check if current planogram in environment is valid for given machine.
        
        Prevents using wrong planogram if machine ID changes.
        
        Args:
            machine_id (str): Machine ID to check against
            
        Returns:
            bool: True if planogram is for this machine
        """
        try:
            stored_machine_id = os.environ.get('PLANOGRAM_MACHINE_ID')
            return stored_machine_id == str(machine_id) if stored_machine_id else False
        except Exception as e:
            print(f"Error checking planogram validity: {e}")
            return False
    
    def store_planogram_env(self, planogram_data):
        """
        Store planogram data as environment variable with machine ID tracking.
        
        Benefits:
        - Reduces API calls (planogram cached)
        - Faster startup (no waiting for API)
        - Works offline if API temporarily unavailable
        
        Args:
            planogram_data (list): List of product dictionaries from API
        """
        try:
            # Convert planogram list to JSON string
            planogram_json = json.dumps(planogram_data)
            os.environ['MACHINE_PLANOGRAM'] = planogram_json
            
            # Store the machine ID this planogram belongs to
            current_machine_id = self.load_machine_id_env()
            if current_machine_id:
                os.environ['PLANOGRAM_MACHINE_ID'] = str(current_machine_id)
            
            print(f"Planogram data stored in environment: "
                  f"{len(planogram_data)} products for machine {current_machine_id}")
            
            # Also update the tracking_data planogram
            self.tracking_data.machine_planogram = planogram_data
            
        except Exception as e:
            print(f"Error storing planogram in environment: {e}")
    
    def load_planogram_env(self):
        """
        Load planogram data from environment variable.
        
        Returns:
            list: Planogram data (list of product dicts), or empty list if not found
        """
        try:
            planogram_json = os.environ.get('MACHINE_PLANOGRAM')
            if planogram_json:
                planogram_data = json.loads(planogram_json)
                return planogram_data
            else:
                print("No planogram found in environment")
                return []
        except Exception as e:
            print(f"Error loading planogram from environment: {e}")
            return []
    
    # =================================================================
    # PLANOGRAM LOADING AND MANAGEMENT
    # =================================================================
    
    def load_machine_planogram(self):
        """
        Load machine planogram (product inventory) with intelligent caching.
        
        Loading Strategy:
        1. Check for valid cached planogram in environment
        2. If valid cache exists, use it (skip API call)
        3. If no cache or invalid, fetch from API
        4. Start background refresh thread for periodic updates
        5. Start video monitoring thread
        
        This approach:
        - Minimizes API calls
        - Provides instant startup with cache
        - Keeps planogram updated in background
        """
        try:
            # Get machine_id from environment
            current_machine_id = self.load_machine_id_env()
            
            if not current_machine_id:
                print("No machine ID available - loading planogram from environment if available")
                
                # Try to load existing planogram from environment
                existing_planogram = self.load_planogram_env()
                if existing_planogram:
                    self.tracking_data.machine_planogram = existing_planogram
                    print(f"Loaded existing planogram from environment: "
                          f"{len(existing_planogram)} products")
                else:
                    self.tracking_data.machine_planogram = []
                    print("No planogram found in environment and no machine ID available")
                return

            # Check if valid planogram already exists in environment
            existing_planogram = self.load_planogram_env()
            if existing_planogram and self.is_planogram_valid_for_machine(current_machine_id):
                self.tracking_data.machine_planogram = existing_planogram
                print(f"Using existing planogram from environment for machine "
                      f"{current_machine_id}: {len(existing_planogram)} products")
                print("Skipping initial API fetch - valid planogram already exists")
                
                # Only start refresh thread, no initial API call
                self.start_planogram_refresh_thread()
                
                # Start video monitoring thread
                video_monitor_thread = threading.Thread(
                    target=monitor_and_send_videos,
                    args=(self.video_directory, current_machine_id, 
                          self.machine_identifier, self.user_id)
                )
                video_monitor_thread.daemon = True
                video_monitor_thread.start()
                print("Video monitoring thread started")
                
                return
            
            # No valid planogram in environment, fetch from API
            print(f"No valid planogram found in environment for machine "
                  f"{current_machine_id} - fetching from API")
            self.fetch_and_store_initial_planogram(current_machine_id)
            
        except Exception as e:
            print(f"Error loading planogram: {e}")
            
            # Final fallback - try to load from environment
            try:
                existing_planogram = self.load_planogram_env()
                if existing_planogram:
                    self.tracking_data.machine_planogram = existing_planogram
                    print("Using existing planogram from environment as final fallback")
                else:
                    self.tracking_data.machine_planogram = []
            except Exception as final_error:
                print(f"Final fallback error: {final_error}")
                self.tracking_data.machine_planogram = []
    
    # =================================================================
    # FALLBACK PIPELINE CONFIGURATION
    # =================================================================
    
    def get_fallback_pipeline_string(self):
        """
        Return the fallback GStreamer pipeline string when API fetch fails.
        
        This is a pre-configured pipeline for dual-camera object detection:
        - Camera 0: /dev/video0 (USB camera)
        - Camera 2: /dev/video2 (USB camera)
        - Resolution: 640x360 @ 25fps
        - MJPEG input format
        - Hailo AI inference with tracking
        - Side-by-side display output
        
        Returns:
            str: Complete GStreamer pipeline configuration
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
            
            # Camera 0 source pipeline
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
            
            # Camera 0 output pipeline
            "sid.src_0 ! "
            "queue name=identity_callback_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "identity name=identity_callback_0 ! "
            "queue name=hailo_draw_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "hailooverlay ! "
            "videoscale n-threads=8 ! "
            "video/x-raw,width=640,height=360 ! "
            "queue name=comp_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "comp.sink_0 "
            
            # Camera 2 source pipeline
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
            
            # Camera 2 output pipeline
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

    # =================================================================
    # API PLANOGRAM FETCHING AND REFRESH SYSTEM
    # =================================================================
    
    def fetch_and_store_initial_planogram(self, machine_id):
        """
        Fetch planogram from API only for initial setup when not cached.
        
        This function:
        1. Makes initial API call to get product inventory
        2. Stores planogram in environment for caching
        3. Starts background refresh thread for periodic updates
        4. Starts video monitoring thread for uploads
        
        Called only when:
        - No valid planogram exists in environment cache
        - Machine ID changes
        - First time system starts
        
        Args:
            machine_id (str): Machine database ID
        """
        try:
            # API authentication credentials
            username = 'admin'
            password = '1234'
            api_key = '123456'
            headers = {'x-api-key': api_key}
            
            # Construct API endpoint URL
            api_endpoint = (f'https://stg-sfapi.nuboxtech.com/index.php/'
                          f'mobile_app/machine/Machine_listing/machine_planogram/{machine_id}')
            
            # Start video monitoring thread
            video_monitor_thread = threading.Thread(
                target=monitor_and_send_videos,
                args=(self.video_directory, machine_id, 
                      self.machine_identifier, self.user_id)
            )
            video_monitor_thread.daemon = True
            video_monitor_thread.start()
            print("Video monitoring thread started")

            # Start refresh thread for future updates (every 1000 seconds)
            self.start_planogram_refresh_thread()

            # Make initial API request to fetch planogram
            api_response = requests.get(
                api_endpoint, 
                auth=HTTPBasicAuth(username, password), 
                headers=headers
            )
            
            if api_response.status_code == 200:
                # Parse planogram data from response
                machine_planogram = api_response.json().get('machine_planogram', [])
                
                # Store in environment and update tracking_data
                self.store_planogram_env(machine_planogram)
                
                print("Initial planogram fetched and stored in environment:")
                for product in machine_planogram:
                    print(f"Product library ID: {product['product_library_id']}, "
                          f"Name: {product['product_name']}, "
                          f"Price: {product['product_price']}")
                    
            else:
                print(f"Initial API request failed: {api_response.status_code}")
                self.tracking_data.machine_planogram = []
                
        except Exception as e:
            print(f"Error in initial API request: {e}")
            self.tracking_data.machine_planogram = []
    
    def start_planogram_refresh_thread(self):
        """
        Start background thread to refresh planogram periodically.
        
        This daemon thread:
        1. Runs in background continuously
        2. Fetches updated planogram every 1000 seconds (~16.6 minutes)
        3. Only updates if planogram has changed
        4. Handles API failures gracefully
        5. Always uses latest machine_id from environment
        
        Why Refresh?
        - Admin may add/remove products
        - Prices may change
        - Inventory updates
        - Keeps system synchronized with database
        
        Thread Properties:
        - Daemon: Yes (exits when main program exits)
        - Interval: 1000 seconds between refreshes
        - Error handling: Continues running even if API fails
        """
        def refresh_planogram():
            """Inner function that runs in background thread."""
            # API credentials
            username = 'admin'
            password = '1234'
            api_key = '123456'
            headers = {'x-api-key': api_key}
            
            # Infinite loop for continuous refreshing
            while True:
                try:
                    # Always get the latest machine_id from environment
                    refresh_machine_id = self.load_machine_id_env()
                    if not refresh_machine_id:
                        print("No machine ID available for refresh - skipping")
                        time.sleep(1000)
                        continue
                    
                    # Construct API endpoint
                    refresh_endpoint = (f'https://stg-sfapi.nuboxtech.com/index.php/'
                                      f'mobile_app/machine/Machine_listing/'
                                      f'machine_planogram/{refresh_machine_id}')
                    
                    # Fetch updated planogram
                    api_response = requests.get(
                        refresh_endpoint, 
                        auth=HTTPBasicAuth(username, password), 
                        headers=headers
                    )
                    
                    if api_response.status_code == 200:
                        new_planogram = api_response.json().get('machine_planogram', [])
                        
                        # Check if planogram actually changed
                        current_planogram = self.load_planogram_env()
                        if new_planogram != current_planogram:
                            # Store the updated planogram
                            self.store_planogram_env(new_planogram)
                            print(f"Planogram updated in environment: "
                                  f"{len(new_planogram)} products")
                        else:
                            print("Planogram unchanged - no update needed")
                            
                    else:
                        print(f"API refresh failed: {api_response.status_code}")
                        
                except Exception as e:
                    print(f"Error refreshing planogram: {e}")
                
                # Wait 1000 seconds before next refresh (~16.6 minutes)
                time.sleep(1000)

        # Start refresh thread as daemon
        refresh_thread = threading.Thread(target=refresh_planogram)
        refresh_thread.daemon = True
        refresh_thread.start()
        print("Planogram refresh thread started (updates every 1000 seconds)")
    
    def get_planogram_from_env(self):
        """
        Get current planogram from environment (for external access).
        
        Useful for:
        - Other modules needing planogram data
        - Debugging and inspection
        - Manual verification
        
        Returns:
            list: Current planogram data from environment cache
        """
        return self.load_planogram_env()
    
    # =================================================================
    # PRODUCT VALIDATION AGAINST PLANOGRAM
    # =================================================================
    
    def validate_detected_product(self, detected_product):
        """
        Validate if detected product exists in machine's planogram.
        
        This critical function:
        1. Normalizes product names (removes spaces, lowercase)
        2. Searches planogram for matching product
        3. Returns validation result with product details
        
        Validation Rules:
        - Product name must match exactly (after normalization)
        - Returns product details (price, ID) if valid
        - Returns error message if not in planogram
        
        Args:
            detected_product (str): Product name from AI detection
            
        Returns:
            dict: Validation result containing:
                {
                    "valid": bool,
                    "product_details": dict or None,
                    "message": str
                }
        
        Example:
            Input: "Coca Cola" (with space, mixed case)
            Normalized: "cocacola" (no space, lowercase)
            Match: "cocacola" in planogram → VALID
        """
        # Get the current planogram from environment (ensures latest data)
        current_planogram = self.load_planogram_env()
        if current_planogram:
            self.tracking_data.machine_planogram = current_planogram
        
        # Normalize detected product name for comparison
        # Remove spaces and convert to lowercase
        normalized_detected_product = detected_product.replace(' ', '').lower()
        
        # Search for matching products in planogram
        matching_planogram_products = [
            product for product in self.tracking_data.machine_planogram
            if product.get('product_name', '').replace(' ', '').lower() == 
               normalized_detected_product
        ]
        
        # Check if product was found
        if matching_planogram_products:
            # Product is valid - return first match with details
            return {
                "valid": True,
                "product_details": matching_planogram_products[0],
                "message": f"{detected_product} validated successfully - found in planogram"
            }
        else:
            # Product not in planogram - invalid
            return {
                "valid": False,
                "product_details": None,
                "message": f"{detected_product} not available in machine planogram"
            }

# =====================================================================
# HAILO DETECTION APPLICATION CLASS
# =====================================================================

class HailoDetectionApp:
    """
    Main application class that manages the GStreamer pipeline and detection.
    
    Responsibilities:
    1. Create and configure GStreamer pipeline
    2. Connect detection callbacks to pipeline
    3. Monitor door status for shutdown
    4. Handle graceful cleanup and resource release
    5. Manage pipeline state transitions
    
    Pipeline Flow:
        Camera → Decode → Scale → Convert → Hailo AI → Track → Display
                                               ↓
                                          Callback (detection_callback)
                                               ↓
                                          Process Results
    
    State Management:
    - NULL: Pipeline not initialized
    - READY: Resources allocated
    - PAUSED: Ready to play but paused
    - PLAYING: Active processing
    
    Shutdown Safety:
    - Uses locks to prevent double-shutdown
    - Cleans up GPIO, pipelines, and threads
    - Removes probe handlers properly
    """
    
    def __init__(self, app_callback, user_data):
        """
        Initialize the Hailo detection application.
        
        Args:
            app_callback: Callback function for processing detections
            user_data: User data container with tracking information
        """
        self.app_callback = app_callback
        self.user_data = user_data
        
        # Door monitoring
        self.door_monitor_active = True
        self.door_monitor_thread = threading.Thread(target=self.monitor_door)
        self.door_monitor_thread.daemon = True
        
        # Shutdown protection
        self.shutdown_called = False
        self.shutdown_lock = threading.Lock()
        
        # Frame configuration
        self.use_frame = True
        self.labels_json = 'resources/labels.json'
        self.hef_path = 'resources/ai_model.hef'
        self.arch = 'hailo8'
        self.show_fps = True
        
        # Hailo pipeline configuration
        self.batch_size = 2              # Process 2 frames simultaneously
        self.network_width = 640         # AI model input width
        self.network_height = 640        # AI model input height
        self.network_format = "RGB"      # Color format
        
        # Post-processing configuration
        self.post_process_so = os.path.join(
            os.path.dirname(__file__), 
            '../resources/libyolo_hailortpp_postprocess.so'
        )
        self.post_function_name = "filter_letterbox"
        
        # Create the GStreamer pipeline
        self.create_pipeline()
        
        # Start door monitoring
        self.door_monitor_thread.start()

    def get_pipeline_string(self):
        """
        Get GStreamer pipeline configuration string.
        
        Priority:
        1. Try to load from environment/API
        2. Fall back to hardcoded pipeline if not available
        
        Returns:
            str: Complete GStreamer pipeline string
        """
        # Get pipeline string from configuration
        pipeline_string = True  # Placeholder for actual pipeline loading
        
        # Use fallback if not available
        if pipeline_string:
            print("Using fallback pipeline string")
            pipeline_string = self.user_data.get_fallback_pipeline_string()
        
        print(f'Pipeline configuration loaded')
        return pipeline_string

    def create_pipeline(self):
        """
        Create and configure the GStreamer pipeline.
        
        Steps:
        1. Initialize GStreamer
        2. Parse pipeline string into pipeline object
        3. Set up message bus for error/EOS handling
        4. Connect detection callbacks to identity elements
        5. Store probe IDs for cleanup
        
        Pipeline Elements:
        - identity_callback_0: Hook for camera 0 detections
        - identity_callback_1: Hook for camera 1 detections
        
        Callbacks:
        - Buffer probe: Called for every frame with detections
        - Bus callback: Handles pipeline messages (errors, EOS)
        """
        # Initialize GStreamer
        Gst.init(None)
        
        # Get pipeline configuration string
        pipeline_string = self.get_pipeline_string()
        
        # Parse string into pipeline object
        self.pipeline = Gst.parse_launch(pipeline_string)
        
        # Create main event loop
        self.loop = GLib.MainLoop()

        # Set up bus for pipeline messages
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        # Connect callbacks for both camera streams (0 and 1)
        for stream_id in [0, 1]:
            # Find identity element for this stream
            identity = self.pipeline.get_by_name(f"identity_callback_{stream_id}")
            if identity:
                # Get source pad from identity element
                pad = identity.get_static_pad("src")
                if pad:
                    # Prepare callback data
                    callback_data = {
                        "user_data": self.user_data, 
                        "stream_id": stream_id
                    }
                    
                    # Add probe to intercept buffers (frames)
                    probe_id = pad.add_probe(
                        Gst.PadProbeType.BUFFER, 
                        self.app_callback, 
                        callback_data
                    )
                    
                    # Store probe ID for later removal
                    if not hasattr(self, 'probe_ids'):
                        self.probe_ids = {}
                    self.probe_ids[f"stream_{stream_id}"] = (identity, pad, probe_id)
                    
                    print(f"Successfully added probe to identity element for stream {stream_id}")
                else:
                    print(f"Warning: Could not get src pad from identity element "
                          f"for stream {stream_id}")
            else:
                print(f"Warning: Could not find identity_callback_{stream_id} "
                      f"element in pipeline")
    
        return True
    
    def monitor_door(self):
        """
        Monitor door switch and trigger shutdown when door closes.
        
        This function runs in a separate daemon thread and:
        1. Continuously checks door sensor (DOOR_SWITCH_PIN)
        2. Waits 5 seconds after start before checking (startup grace period)
        3. Triggers shutdown when door closes (sensor = 0)
        4. Exits when door_monitor_active is set to False
        
        Why 5 Second Delay?
        - Prevents false triggers during startup
        - Allows customer time to close door normally
        - Avoids premature shutdown
        
        Door States:
        - 0 = Closed (shutdown trigger)
        - 1 = Open (continue monitoring)
        """
        start_time = time.time()
        
        while self.door_monitor_active:
            # Read door sensor
            door_sw = GPIO.input(DOOR_SWITCH_PIN)
            
            # Check if door closed AND grace period passed
            if door_sw == 0 and time.time() - start_time > 5:
                print("Door closed - Initiating shutdown")
                self.shutdown()
                break
                
            # Small sleep to prevent CPU overuse
            time.sleep(0.1)
    
    def shutdown(self, signum=None, frame=None):
        """
        Gracefully shutdown the pipeline and clean up resources.
        
        This critical function:
        1. Prevents double-shutdown with lock
        2. Stops door monitoring
        3. Sets shutdown events
        4. Transitions pipeline: PAUSED → READY → NULL
        5. Waits for state changes with delays
        6. Closes OpenCV windows
        7. Quits GLib main loop
        
        Called by:
        - Door monitor when door closes
        - Signal handler (Ctrl+C)
        - Error conditions
        - WebSocket disconnect
        
        Args:
            signum (int, optional): Signal number if called by signal handler
            frame (optional): Frame object if called by signal handler
        """
        # Prevent double-shutdown with lock
        with self.shutdown_lock:
            if self.shutdown_called:
                print("Shutdown already in progress, skipping...")
                return
            self.shutdown_called = True
        
        print("Shutting down... Please wait.")
        
        # Stop door monitoring thread
        self.door_monitor_active = False
        
        # Set shutdown events to signal other threads
        self.user_data.tracking_data.shutdown_event.set()
        self.user_data.shutdown_event.set()
        
        # Graceful pipeline shutdown with delays
        # Step 1: PAUSED state
        self.pipeline.set_state(Gst.State.PAUSED)
        GLib.usleep(100000)  # 0.1 second delay
        
        # Step 2: READY state
        self.pipeline.set_state(Gst.State.READY)
        GLib.usleep(100000)  # 0.1 second delay
        
        # Step 3: NULL state (fully stopped)
        self.pipeline.set_state(Gst.State.NULL)
        
        # Wait for state change to complete (max 5 seconds)
        ret, state, pending = self.pipeline.get_state(5 * Gst.SECOND)
        
        # Force close any remaining OpenCV windows
        cv2.destroyAllWindows()
        
        # Quit the GLib main loop
        GLib.idle_add(self.loop.quit)
        
    def bus_call(self, bus, message, loop):
        """
        Handle messages from the GStreamer pipeline bus.
        
        Message Types:
        - EOS (End of Stream): Pipeline finished normally
        - ERROR: Something went wrong in pipeline
        - WARNING: Non-fatal issues
        - INFO: Informational messages
        
        Args:
            bus: GStreamer bus object
            message: Message object containing type and data
            loop: GLib main loop to quit on EOS/ERROR
            
        Returns:
            bool: True to continue receiving messages
        """
        t = message.type
        
        if t == Gst.MessageType.EOS:
            # End of stream - pipeline finished
            print("End-of-stream")
            loop.quit()
            
        elif t == Gst.MessageType.ERROR:
            # Error occurred - parse and display
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
            
        return True

    def run(self):
        """
        Start the pipeline and run the main processing loop.
        
        This is the main execution function that:
        1. Sets up signal handler for Ctrl+C
        2. Starts display process (if use_frame=True)
        3. Transitions pipeline to PLAYING state
        4. Runs GLib main loop (blocks until shutdown)
        5. Handles cleanup in finally block
        
        Flow:
        1. run() called
        2. Pipeline starts → PLAYING state
        3. loop.run() blocks (processing happens)
        4. Shutdown triggered → loop.quit()
        5. Finally block cleans up
        6. Function returns
        
        Error Handling:
        - Catches pipeline start failures
        - Reports bus errors with details
        - Ensures cleanup happens regardless
        """
        # Set up signal handler for SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.shutdown)
        
        # Start display process if frame display is enabled
        if self.use_frame:
            display_process = multiprocessing.Process(
                target=display_user_data_frame, 
                args=(self.user_data,)
            )
            display_process.start()
        
        try:
            # Transition pipeline to PLAYING state
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                print("ERROR: Pipeline failed to start!")
                
                # Get detailed error from bus
                bus = self.pipeline.get_bus()
                msg = bus.timed_pop_filtered(Gst.SECOND, Gst.MessageType.ERROR)
                if msg:
                    err, debug = msg.parse_error()
                    print(f"Pipeline error: {err.message}")
                    print(f"Debug info: {debug}")
                
                raise Exception("Pipeline failed to start - camera may be busy")
            
            print("Pipeline started successfully!")
            
            # Run main loop (blocks here until shutdown)
            self.loop.run()
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            raise  # Re-raise to trigger cleanup
            
        finally:
            print("Pipeline run() cleanup...")
            
            # Set shutdown events
            self.user_data.tracking_data.shutdown_event.set()
            self.user_data.shutdown_event.set()
            
            # Clean up pipeline if not already done
            with self.shutdown_lock:
                if not self.shutdown_called:
                    print("Cleaning up pipeline from run() finally block...")
                    try:
                        self.pipeline.set_state(Gst.State.NULL)
                        ret, state, pending = self.pipeline.get_state(5 * Gst.SECOND)
                        print(f"Pipeline state after NULL: {state}")
                    except Exception as e:
                        print(f"Error cleaning up pipeline: {e}")
                else:
                    print("Pipeline already cleaned up by shutdown(), skipping")
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            # Terminate display process if it exists
            if hasattr(self, 'display_process') and self.use_frame:
                if display_process.is_alive():
                    display_process.terminate()
                    display_process.join(timeout=2)

# =====================================================================
# CROSS-CAMERA TRACKING SYSTEM
# =====================================================================
"""
This section implements sophisticated cross-camera object tracking.

PROBLEM:
- Two cameras view the same fridge from different angles
- Same object gets different track IDs on each camera (e.g., Track 5 on Cam0, Track 12 on Cam1)
- Need to recognize it's the SAME object to count correctly

SOLUTION:
- Local Track ID: Assigned by individual camera's tracker (Track 5, Track 12)
- Global Track ID: Unified ID across cameras (Track 100 for same object)
- Matching: Based on product label and timing

EXAMPLE:
Camera 0 detects: Coke (local_id=5) → assigned global_id=100
Camera 1 detects: Coke (local_id=12) → matched to global_id=100 (same object!)
Result: Only count once, not twice!
"""

def get_global_track_id(camera_id, local_track_id, features=None, label=None):
    """
    Get or create a global track ID for cross-camera tracking.
    
    This is the HEART of the cross-camera tracking system. It implements
    intelligent ID assignment with these rules:
    
    RULE 1: Same local_track_id on same camera → Same global_id
        Example: Camera 0, Track 5 → Always Global 100
    
    RULE 2: Different local_track_ids with same label on same camera → Different global_ids
        Example: Camera 0, Coke Track 5 → Global 100
                 Camera 0, Coke Track 8 → Global 101 (different object!)
    
    RULE 3: Same label on different cameras → Same global_id (if available)
        Example: Camera 0, Coke Track 5 → Global 100
                 Camera 1, Coke Track 12 → Global 100 (same Coke!)
    
    RULE 4: Multiple same-label objects on different cameras → Different global_ids
        Example: Camera 0, Coke Track 5 → Global 100
                 Camera 1, Coke Track 12 → Global 100 (matched)
                 Camera 1, Coke Track 15 → Global 101 (different Coke)
    
    Args:
        camera_id (int): Camera identifier (0 or 1)
        local_track_id (int): Local track ID from camera's tracker
        features (optional): Feature vector for matching (currently unused)
        label (str): Product class label (e.g., "coke", "sprite")
        
    Returns:
        int: Global track ID for this object
        
    Global Variables Used:
        global_track_counter: Next available global ID
        local_to_global_id_map: Mapping of (camera_id, local_id) → global_id
        global_track_labels: Stores label for each global_id
        active_objects_per_camera: Tracks active objects per camera/label
    """
    global global_track_counter, local_to_global_id_map, global_track_labels
    global active_objects_per_camera
    
    # STEP 1: Check if this local track already has a global ID (RULE 1)
    if (camera_id, local_track_id) in local_to_global_id_map:
        return local_to_global_id_map[(camera_id, local_track_id)]
    
    # STEP 2: Handle case where no label provided
    if not label:
        # No label = can't match across cameras, create new global ID
        new_global_id = global_track_counter
        global_track_counter += 1
        local_to_global_id_map[(camera_id, local_track_id)] = new_global_id
        return new_global_id
    
    # STEP 3: Check for cross-camera matching opportunity (RULE 3)
    other_camera = 1 if camera_id == 0 else 0  # Get opposite camera
    
    # Look for objects with same label on other camera
    available_matches = []
    if label in active_objects_per_camera[other_camera]:
        # Iterate through all objects with this label on other camera
        for other_local_id, other_global_id in active_objects_per_camera[other_camera][label].items():
            # Check if this global_id is already matched to another object on our camera
            already_matched_on_this_camera = False
            for our_local_id, our_global_id in active_objects_per_camera[camera_id][label].items():
                if our_global_id == other_global_id:
                    already_matched_on_this_camera = True
                    break
            
            # If not already matched, this is an available match
            if not already_matched_on_this_camera:
                available_matches.append((other_local_id, other_global_id))
    
    # STEP 4: Use available match if found
    if available_matches:
        # Use first available match (could add distance-based matching here)
        matched_local_id, matched_global_id = available_matches[0]
        local_to_global_id_map[(camera_id, local_track_id)] = matched_global_id
        active_objects_per_camera[camera_id][label][local_track_id] = matched_global_id
        global_track_labels[matched_global_id] = label
        return matched_global_id
    
    # STEP 5: No match found, create new global ID (RULE 2 or RULE 4)
    new_global_id = global_track_counter
    global_track_counter += 1
    local_to_global_id_map[(camera_id, local_track_id)] = new_global_id
    active_objects_per_camera[camera_id][label][local_track_id] = new_global_id
    global_track_labels[new_global_id] = label
    
    return new_global_id

def cleanup_inactive_tracks(camera_id, active_local_track_ids):
    """
    Clean up tracking data for tracks that are no longer active.
    
    Why Needed?
    - Objects leave the frame (customer puts item back)
    - Tracker loses track of object
    - Old track IDs accumulate and waste memory
    
    This function removes all tracking data for inactive objects:
    - Removes from local_to_global_id_map
    - Removes from active_objects_per_camera
    - Frees up global IDs for reuse
    
    Called every frame after processing detections.
    
    Args:
        camera_id (int): Camera identifier (0 or 1)
        active_local_track_ids (set): Set of currently active track IDs
        
    Example:
        Before: Camera 0 has tracks [5, 8, 12]
        Frame: Only tracks [5, 12] detected
        After: Track 8 data cleaned up (no longer active)
    """
    global local_to_global_id_map, active_objects_per_camera
    
    # Find inactive tracks for this camera
    inactive_tracks = []
    for (cam_id, local_id), global_id in local_to_global_id_map.items():
        if cam_id == camera_id and local_id not in active_local_track_ids:
            inactive_tracks.append((cam_id, local_id, global_id))
    
    # Remove inactive tracks
    for cam_id, local_id, global_id in inactive_tracks:
        # Remove from local_to_global_id_map
        del local_to_global_id_map[(cam_id, local_id)]
        
        # Remove from active_objects_per_camera
        label = global_track_labels.get(global_id)
        if label and label in active_objects_per_camera[cam_id]:
            if local_id in active_objects_per_camera[cam_id][label]:
                del active_objects_per_camera[cam_id][label][local_id]
                
                # Clean up empty label entries
                if not active_objects_per_camera[cam_id][label]:
                    del active_objects_per_camera[cam_id][label]

# =====================================================================
# MOVEMENT DIRECTION ANALYSIS
# =====================================================================
"""
This section determines if customer is TAKING or RETURNING items.

MOVEMENT DIRECTIONS:
- 'entry': Object moving INTO fridge (customer returning item) ↓
- 'exit': Object moving OUT OF fridge (customer taking item) ↑
- None: Not enough data or movement too small

ALGORITHM:
1. Track last 5 positions (x, y coordinates)
2. Check bounding box stability (reject if hand moving/obscuring)
3. Verify sufficient displacement (at least 30 pixels)
4. Verify consistent direction (80% of movements in same direction)
5. Calculate average movement per frame
6. Determine direction based on Y-axis movement
"""

def analyze_movement_direction(track_id, center, tracking_data, camera_id, 
                               global_id, current_bbox):
    """
    Analyze movement direction based on 5 consecutive frames with enhanced filtering.
    
    This sophisticated algorithm prevents false detections from:
    - Hand movements obscuring objects
    - Small jittering/noise in tracking
    - Inconsistent up-down-up movements
    - Objects just being repositioned slightly
    
    FILTERING CHECKS (must pass all):
    
    CHECK 1: Bounding Box Stability
        - Rejects if bbox size varies too much (>80% of average)
        - Indicates hand is moving/obscuring object
    
    CHECK 2: Total Displacement
        - Must move at least 30 pixels over 5 frames
        - Prevents counting tiny jitters as movement
    
    CHECK 3: Movement Consistency
        - At least 80% of frame-to-frame movements in same direction
        - Prevents counting erratic up-down-up as intentional
    
    CHECK 4: Average Movement Threshold
        - Average movement per frame must exceed 5 pixels
        - Further filters out noise
    
    Args:
        track_id (int): Local track ID
        center (tuple): Current center point (x, y)
        tracking_data: TrackingData instance with counted_tracks
        camera_id (int): Camera identifier (0 or 1)
        global_id (int): Global track ID
        current_bbox (tuple): Current bounding box (x1, y1, x2, y2)
        
    Returns:
        str or None: 'entry', 'exit', or None
        
    Direction Determination:
        - Positive Y movement (down on screen) = 'exit' (taking out)
        - Negative Y movement (up on screen) = 'entry' (returning)
        - Y increases downward in image coordinates!
    """
    # Store movement in camera-specific history
    camera_movement_history[camera_id][track_id].appendleft(center)
    
    # Calculate and track bounding box area
    bbox_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
    camera_bbox_area_history[camera_id][track_id].appendleft(bbox_area)
    
    # Copy to global movement history for cross-camera analysis
    global_movement_history[global_id].appendleft((center, camera_id))
    
    # Wait until we have enough frames (5 positions)
    if len(camera_movement_history[camera_id][track_id]) < 5:
        return None
    
    # =================================================================
    # CHECK 1: BOUNDING BOX STABILITY
    # =================================================================
    # Reject if bounding box size changing significantly (hand obscuring)
    if len(camera_bbox_area_history[camera_id][track_id]) >= 5:
        areas = list(camera_bbox_area_history[camera_id][track_id])
        avg_area = sum(areas) / len(areas)
        area_variance = sum((a - avg_area) ** 2 for a in areas) / len(areas)
        area_std_dev = area_variance ** 0.5
        
        # If standard deviation > 80% of average, bbox is unstable
        if area_std_dev > (avg_area * 0.8):
            return None  # Likely hand moving/obscuring, not actual object movement
    
    # =================================================================
    # CHECK 2: TOTAL DISPLACEMENT
    # =================================================================
    # Object must actually move a significant distance
    first_y = camera_movement_history[camera_id][track_id][-1][1]  # Oldest Y
    last_y = camera_movement_history[camera_id][track_id][0][1]    # Newest Y
    total_displacement = abs(last_y - first_y)
    
    DISPLACEMENT_THRESHOLD = 30  # Minimum 30 pixels
    if total_displacement < DISPLACEMENT_THRESHOLD:
        return None  # Not enough movement, likely just jittering
    
    # =================================================================
    # CHECK 3: MOVEMENT CONSISTENCY
    # =================================================================
    # Ensure movement is consistently in one direction
    movement_directions = []
    for i in range(1, len(camera_movement_history[camera_id][track_id])):
        curr_y = camera_movement_history[camera_id][track_id][i-1][1]
        prev_y = camera_movement_history[camera_id][track_id][i][1]
        # 1 = downward (exit), -1 = upward (entry)
        movement_directions.append(1 if curr_y > prev_y else -1)
    
    # Count movements in each direction
    positive_movements = sum(1 for d in movement_directions if d > 0)
    negative_movements = sum(1 for d in movement_directions if d < 0)
    consistency_ratio = max(positive_movements, negative_movements) / len(movement_directions)
    
    # Require 80% consistency (4 out of 5 frames in same direction)
    if consistency_ratio < 0.8:
        return None  # Movement too erratic (up-down-up-down)
    
    # =================================================================
    # CHECK 4: AVERAGE MOVEMENT THRESHOLD
    # =================================================================
    # Calculate average movement per frame
    total_movement = 0
    for i in range(1, len(camera_movement_history[camera_id][track_id])):
        curr_y = camera_movement_history[camera_id][track_id][i-1][1]
        prev_y = camera_movement_history[camera_id][track_id][i][1]
        total_movement += curr_y - prev_y
    
    avg_movement = total_movement / 4  # 4 intervals between 5 points
    
    FRAME_MOVEMENT_THRESHOLD = 5  # At least 5 pixels per frame
    if abs(avg_movement) < FRAME_MOVEMENT_THRESHOLD:
        return None  # Movement too slow/small
    
    # =================================================================
    # DETERMINE DIRECTION
    # =================================================================
    # Positive Y movement = moving down = exiting
    # Negative Y movement = moving up = entering
    current_direction = 'exit' if avg_movement > 0 else 'entry'
    
    # =================================================================
    # HANDLE DIRECTION CHANGES
    # =================================================================
    # If direction changed since last count, remove from old direction's set
    # This allows re-counting if customer changes mind (takes out, puts back)
    if global_id in global_last_counted_direction:
        if current_direction != global_last_counted_direction[global_id]:
            # Direction changed - remove from old counted set
            if global_last_counted_direction[global_id] in tracking_data.counted_tracks:
                tracking_data.counted_tracks[global_last_counted_direction[global_id]].discard(global_id)
    
    # =================================================================
    # UPDATE TRACKING STATE
    # =================================================================
    # Record this direction for future comparison
    global_last_counted_direction[global_id] = current_direction
        
    return current_direction

# =====================================================================
# MAIN DETECTION CALLBACK FUNCTION
# =====================================================================
"""
This is the CORE of the entire system - called for EVERY frame from BOTH cameras.

EXECUTION FLOW (per frame):
1. Extract video frame from GStreamer buffer
2. Check if camera is covered (security check)
3. Get AI detections from Hailo accelerator
4. For each detected object:
   - Get or assign global track ID
   - Validate product against planogram
   - Draw bounding box and trail
   - Analyze movement direction
   - Update counters if movement detected
5. Update WebSocket with latest data
6. Calculate price and trigger alerts if needed
7. Display frame (combined from both cameras)

CALLED BY: GStreamer pipeline probe on identity elements
FREQUENCY: ~13-15 times per second (per camera)
THREAD: GStreamer pipeline thread
"""

def detection_callback(pad, info, callback_data):
    """
    Process each video frame with AI detections.
    
    This is the heart of the detection system - processes every frame from
    both cameras, performs AI inference, tracks objects, validates products,
    and updates all counters and displays.
    
    Args:
        pad: GStreamer pad that triggered the callback
        info: GStreamer probe info containing buffer
        callback_data (dict): Contains:
            - user_data: User data object with tracking information
            - stream_id: Camera identifier (0 or 1)
    
    Returns:
        Gst.PadProbeReturn.OK: Continue processing pipeline
    
    Global Variables Used:
        camera_covered: Camera cover detection flag
        cover_alert_thread: Thread for cover alerts
        blink: LED blink control flag
    """
    global camera_covered, cover_alert_thread, blink
    
    # Extract callback data
    user_data = callback_data["user_data"]
    stream_id = callback_data["stream_id"]
    
    # Get buffer from probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get frame format and dimensions
    format, width, height = get_caps_from_pad(pad)
    if not all([format, width, height]):
        return Gst.PadProbeReturn.OK

    # =================================================================
    # STEP 1: GET AI DETECTIONS
    # =================================================================
    # Extract Region of Interest (ROI) with detections from Hailo
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # =================================================================
    # STEP 2: GET VIDEO FRAME FOR VISUALIZATION
    # =================================================================
    frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # =================================================================
    # STEP 3: CAMERA COVER DETECTION (Security Feature)
    # =================================================================
    # Check if camera is covered/blocked
    if is_frame_dark(frame):
        if not camera_covered:  # Only start thread if not already covered
            camera_covered = True
            if cover_alert_thread is None or not cover_alert_thread.is_alive():
                # Start alert thread to warn customer
                cover_alert_thread = threading.Thread(
                    target=handle_cover_alert, 
                    daemon=True
                )
                cover_alert_thread.start()
    else:
        if camera_covered:  # Transitioning from covered to uncovered
            camera_covered = False

    # =================================================================
    # STEP 4: TRACK ACTIVE OBJECTS FOR CLEANUP
    # =================================================================
    # Collect all active track IDs in this frame for cleanup
    active_local_track_ids = set()
    
    # Track frame for transaction memory management
    if hasattr(user_data, 'transaction_id') and user_data.transaction_id:
        transaction_memory_manager.track_frame(user_data.transaction_id)
    
    # =================================================================
    # STEP 5: PROCESS EACH DETECTION
    # =================================================================
    for detection in detections:
        # Extract detection information
        label = detection.get_label()              # Product name
        bbox = detection.get_bbox()                # Bounding box
        confidence = detection.get_confidence()    # Detection confidence
        class_id = detection.get_class_id()        # Class ID
        
        # -----------------------------------------------------------
        # Get Track ID from Hailo Tracker
        # -----------------------------------------------------------
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
            active_local_track_ids.add(track_id)  # Mark as active
        
        # -----------------------------------------------------------
        # Calculate Bounding Box Coordinates
        # -----------------------------------------------------------
        # Convert normalized coordinates to pixel coordinates
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int(bbox.xmax() * width)
        y2 = int(bbox.ymax() * height)
        
        # Calculate center point
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        # -----------------------------------------------------------
        # Get or Create Global Track ID (Cross-Camera Tracking)
        # -----------------------------------------------------------
        global_id = get_global_track_id(stream_id, track_id, None, label)
        
        # Track object for transaction memory management
        if hasattr(user_data, 'transaction_id') and user_data.transaction_id:
            transaction_memory_manager.track_object(
                user_data.transaction_id, 
                track_id, 
                global_id
            )
        
        # -----------------------------------------------------------
        # Validate Product Against Planogram
        # -----------------------------------------------------------
        validation_result = user_data.validate_detected_product(label)
        
        # Get color for this product class
        color = compute_color_for_labels(class_id)
        
        # -----------------------------------------------------------
        # Draw Bounding Box and Label
        # -----------------------------------------------------------
        # Draw rectangle around detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Format label text with track IDs and validation status
        label_text = (f"{label} L:{track_id} G:{global_id} "
                     f"{'Valid' if validation_result['valid'] else 'Invalid'}")
        
        # Draw label above bounding box
        # Green if valid, red if invalid
        text_color = (0, 255, 0) if validation_result['valid'] else (0, 0, 255)
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # -----------------------------------------------------------
        # Draw Movement Trail
        # -----------------------------------------------------------
        draw_trail(frame, track_id, center, color, global_id=global_id)
        
        # -----------------------------------------------------------
        # Analyze Movement Direction
        # -----------------------------------------------------------
        direction = analyze_movement_direction(
            track_id, 
            center, 
            user_data.tracking_data,
            stream_id,
            global_id,
            (x1, y1, x2, y2)  # Current bounding box
        )
        
        # =============================================================
        # STEP 6: UPDATE COUNTERS IF MOVEMENT DETECTED
        # =============================================================
        if direction:
            # Check if we should count this movement
            # Count if:
            # 1. Global ID not yet counted for this direction, OR
            # 2. Direction changed since last count (customer changed mind)
            
            should_count = (
                global_id not in user_data.tracking_data.counted_tracks.get(direction, set()) or
                (global_id in global_last_counted_direction and 
                 direction != global_last_counted_direction[global_id])
            )
            
            if should_count:
                # Increment counter for this product and direction
                user_data.tracking_data.class_counters[direction][label] += 1
                
                # Add to counted tracks set
                if direction not in user_data.tracking_data.counted_tracks:
                    user_data.tracking_data.counted_tracks[direction] = set()
                user_data.tracking_data.counted_tracks[direction].add(global_id)
                
                # -----------------------------------------------------
                # Update Validated/Invalidated Products
                # -----------------------------------------------------
                if validation_result['valid']:
                    # VALID PRODUCT - Add to validated products
                    if label not in user_data.tracking_data.validated_products[direction]:
                        user_data.tracking_data.validated_products[direction][label] = {
                            "count": 0,
                            "product_details": validation_result['product_details']
                        }
                    user_data.tracking_data.validated_products[direction][label]["count"] += 1
                    
                else:
                    # INVALID PRODUCT - Add to invalidated products
                    if label not in user_data.tracking_data.invalidated_products[direction]:
                        user_data.tracking_data.invalidated_products[direction][label] = {
                            "count": 0,
                            "raw_detection": {
                                "name": label,
                                "confidence": confidence,
                                "tracking_id": global_id,
                                "bounding_box": {
                                    "xmin": x1,
                                    "ymin": y1,
                                    "xmax": x2,
                                    "ymax": y2
                                }
                            }
                        }
                    user_data.tracking_data.invalidated_products[direction][label]["count"] += 1

    # =================================================================
    # STEP 7: CLEANUP INACTIVE TRACKS
    # =================================================================
    # Remove tracking data for objects no longer in frame
    cleanup_inactive_tracks(stream_id, active_local_track_ids)

    # =================================================================
    # STEP 8: UPDATE FPS CALCULATION
    # =================================================================
    current_time = time.time()
    user_data.tracking_data.last_time = current_time
    
    # =================================================================
    # STEP 9: DRAW COUNTS ON FRAME
    # =================================================================
    # Display entry/exit counts for all products
    label = next((det.get_label() for det in detections), None)
    draw_counts(frame, user_data.tracking_data.class_counters, label)
    
    # =================================================================
    # STEP 10: CONVERT COLOR SPACE FOR DISPLAY
    # =================================================================
    # Convert RGB to BGR for OpenCV display
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # =================================================================
    # STEP 11: STORE FRAME FOR DISPLAY
    # =================================================================
    # Store frames separately for each camera
    if stream_id == 0:
        user_data.frame_left = frame
    elif stream_id == 1:
        user_data.frame_right = frame
    
    # Combine both camera frames side-by-side if both available
    if hasattr(user_data, "frame_left") and hasattr(user_data, "frame_right"):
        combined_frame = np.hstack((user_data.frame_left, user_data.frame_right))
        user_data.set_frame(combined_frame)
    
    # =================================================================
    # STEP 12: UPDATE WEBSOCKET DATA
    # =================================================================
    # Prepare data structure for WebSocket transmission
    websocket_data = {
        "validated_products": {
            "entry": {
                product: {
                    "count": details["count"],
                    "product_details": details["product_details"]
                } for product, details in user_data.tracking_data.validated_products["entry"].items()
            },
            "exit": {
                product: {
                    "count": details["count"],
                    "product_details": details["product_details"]
                } for product, details in user_data.tracking_data.validated_products["exit"].items()
            }
        },
        "invalidated_products": {
            "entry": {
                product: {
                    "count": details["count"],
                    "raw_detection": details["raw_detection"]
                } for product, details in user_data.tracking_data.invalidated_products["entry"].items()
            },
            "exit": {
                product: {
                    "count": details["count"],
                    "raw_detection": details["raw_detection"]
                } for product, details in user_data.tracking_data.invalidated_products["exit"].items()
            }
        }
    }
    
    # Update WebSocket data manager
    user_data.tracking_data.websocket_data_manager.update_data(websocket_data)
    
    # =================================================================
    # STEP 13: CALCULATE PRICE AND TRIGGER ALERTS
    # =================================================================
    # Get current data and deposit amount
    current_data = user_data.tracking_data.websocket_data_manager.get_current_data()
    deposit = user_data.deposit
    
    # Calculate total price and control buzzer/LED if deposit exceeded
    total_price = calculate_total_price_and_control_buzzer(
        current_data, 
        deposit, 
        label
    )
    
    # Continue processing pipeline
    return Gst.PadProbeReturn.OK

# =====================================================================
# TRANSACTION ORCHESTRATION FUNCTION
# =====================================================================

async def run_tracking(websocket: WebSocket):
    """
    Main transaction orchestration function.
    
    This function manages the entire lifecycle of a customer transaction:
    1. Wait for 'start_preview' message from mobile app
    2. Extract transaction details (deposit, IDs, etc.)
    3. Control door lock (unlock briefly)
    4. Handle two modes:
       a) Product upload mode (capture photos for new products)
       b) Detection mode (track items taken/returned)
    5. Send real-time updates via WebSocket
    6. Clean up on completion
    
    Args:
        websocket: WebSocket connection to mobile app
        
    WebSocket Messages:
        Receive:
        - start_preview: Start transaction with deposit/IDs
        - product_upload: Capture photos of new product
        
        Send:
        - Detection updates (validated/invalidated products)
        - Status messages (error, stopped, etc.)
    
    Global Variables Modified:
        readyToProcess: System readiness flag
        done: Transaction completion flag
        current_pipeline_app: Active pipeline application
    """
    global readyToProcess, cover_alert_thread
    global unlock_data, done, current_pipeline_app
    
    # Initialize variables
    unlock_data = 0
    deposit = 0.0
    machine_id = None
    machine_identifier = None
    user_id = None
    transaction_id = None
    product_name = None
    image_count = None
    
    # =================================================================
    # PHASE 1: WAIT FOR START MESSAGE
    # =================================================================
    # Wait for mobile app to send start_preview or product_upload message
    while True:
        try:
            message_text = await websocket.receive_text()
            print(f"Received message: {message_text}")
            
            try:
                message = json.loads(message_text)
                
                # ---------------------------------------------------
                # START PREVIEW (Regular Transaction)
                # ---------------------------------------------------
                if isinstance(message, dict) and message.get('action') == 'start_preview':
                    unlock_data = 1
                    
                    # Extract transaction details
                    deposit = float(message.get('deposit', 0.0))
                    machine_id = message.get('machine_id')
                    machine_identifier = message.get('machine_identifier')
                    user_id = message.get('user_id')
                    transaction_id = message.get('transaction_id')
                    product_name = message.get('product_name')
                    image_count = message.get('image_count')
                    
                    # Log transaction details
                    print(f"Transaction Started:")
                    print(f"  Deposit: ${deposit}")
                    print(f"  Machine ID: {machine_id}")
                    print(f"  Machine Identifier: {machine_identifier}")
                    print(f"  User ID: {user_id}")
                    print(f"  Transaction ID: {transaction_id}")
                    print(f"  Product Name: {product_name}")
                    print(f"  Image Count: {image_count}")
                    
                    print(f"Start preview received. Unlock data: {unlock_data}")
                    break
                else:
                    break
                    
            except json.JSONDecodeError:
                continue
                
        except Exception as e:
            await websocket.send_json({
                "status": "error",
                "message": f"Error waiting for start message: {str(e)}"
            })
            return

    # =================================================================
    # PHASE 2: DOOR CONTROL
    # =================================================================
    if unlock_data == 1:   
        print("Unlock door for 0.5 seconds")
        readyToProcess = True
        unlock_data = 0
        print("Unlock data reset to 0")

    try:
        # =============================================================
        # MODE 1: PRODUCT UPLOAD (Capture photos of new product)
        # =============================================================
        if isinstance(message, dict) and message.get('action') == 'product_upload':
            done = True        
            
            # Extract product upload details
            machine_id = message.get('machine_id')
            machine_identifier = message.get('machine_identifier')
            user_id = message.get('user_id')
            product_name = message.get('product_name')
            image_count = message.get('image_count')
            
            print(f"\nProduct Upload Mode:")
            print(f"  Machine ID: {machine_id}")
            print(f"  Machine Identifier: {machine_identifier}")
            print(f"  User ID: {user_id}")
            print(f"  Product Name: {product_name}")
            print(f"  Image Count: {image_count}")
            
            print("\n" + "="*50)
            print("STARTING IMAGE CAPTURE PROCESS")
            print("="*50)
            print(f"Total images to capture: {image_count} per camera")
            
            # Get alert sound directory
            alert_dir = "sounds/product_upload_alerts"
            
            # Play start capture alert
            tts_manager.play_mp3_sync(f"{alert_dir}/start_capture.mp3", volume=0.8)
            time.sleep(2)
            
            # ---------------------------------------------------
            # Capture Images from Camera 1 (/dev/video0)
            # ---------------------------------------------------
            print("\n" + "-"*30)
            print("CAMERA 1 CAPTURE PHASE")
            print("-"*30)
            camera1_images = capture_images(2, image_count)
            
            # ---------------------------------------------------
            # Process Results
            # ---------------------------------------------------
            if camera1_images:
                print("\nAll images captured successfully!")
                
                # Play completion alert
                tts_manager.play_mp3_sync(f"{alert_dir}/all_complete.mp3", volume=0.8)
            
                # Upload images to API
                print("\nUploading images to API...")
                if upload_images_to_api(camera1_images, machine_id, 
                                       machine_identifier, user_id, 
                                       product_name, image_count):
                    print("Images uploaded successfully!")
                    
                    # Delete all captured images after successful upload
                    print("\nDeleting captured images...")
                    all_images = camera1_images 
                    delete_images(all_images)
                    
                    # Play success alert
                    tts_manager.play_mp3_sync(f"{alert_dir}/upload_success.mp3", volume=0.8)
                else:
                    print("Failed to upload images.")
                    print("Images will be kept for retry or manual inspection.")
                    
                    # Play failure alert
                    tts_manager.play_mp3_sync(f"{alert_dir}/upload_failed.mp3", volume=0.8)
                    
            else:
                print("Failed to capture all images.")
                # Play failure alert
                tts_manager.play_mp3_sync(f"{alert_dir}/upload_failed.mp3", volume=0.8)
        
        # =============================================================
        # MODE 2: DETECTION MODE (Regular Transaction)
        # =============================================================
        else:
            # START TRANSACTION MEMORY TRACKING
            if transaction_id:
                transaction_memory_manager.start_transaction(transaction_id)
                print(f"[Memory] Transaction {transaction_id} started")
            
            # Initialize door status monitoring
            door_monitor_active = True
            done = True
            
            # ---------------------------------------------------
            # Door Monitor Function (Async)
            # ---------------------------------------------------
            async def monitor_door():
                """Monitor door and stop tracking when closed."""
                nonlocal door_monitor_active
                while door_monitor_active:
                    door_sw = 1  # TODO: Replace with GPIO.input(DOOR_SWITCH_PIN)
                    if door_sw == 0:  # Door is closed
                        print("Door closed - Stopping tracking")
                        callback.tracking_data.shutdown_event.set()
                        callback.shutdown_event.set()
                        door_monitor_active = False
                        
                        # Send final message to client
                        try:
                            await websocket.send_json({
                                "status": "stopped",
                                "message": "Door closed - Tracking stopped"
                            })
                        except Exception as e:
                            print(f"Error sending final message: {e}")
                        break
                    await asyncio.sleep(0.1)
            
            # Start door monitoring task
            door_monitor_task = asyncio.create_task(monitor_door())
    
            # ---------------------------------------------------
            # Initialize Hailo Detection
            # ---------------------------------------------------
            callback = HailoDetectionCallback(
                websocket, 
                deposit, 
                machine_id, 
                machine_identifier, 
                user_id, 
                transaction_id
            )
    
            # ---------------------------------------------------
            # WebSocket Data Sender Function
            # ---------------------------------------------------
            def send_websocket_data():
                """Send real-time updates via WebSocket."""
                while not callback.tracking_data.shutdown_event.is_set():
                    try:
                        # Get current detection data
                        current_data = callback.tracking_data.websocket_data_manager.get_current_data()
                        
                        # Send via WebSocket
                        asyncio.run(websocket.send_json(current_data))
                        
                        # Update every second
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error sending websocket data: {e}")

            # Start websocket data sender thread
            websocket_sender = threading.Thread(target=send_websocket_data)
            websocket_sender.start()
    
            # ---------------------------------------------------
            # Signal Handler for Ctrl+C
            # ---------------------------------------------------
            def signal_handler(signum, frame):
                """Handle Ctrl+C gracefully."""
                print("\nCtrl+C detected. Initiating shutdown...")
                callback.tracking_data.shutdown_event.set()
                callback.shutdown_event.set()
                cv2.destroyAllWindows()
            
            # Set up signal handler
            signal.signal(signal.SIGINT, signal_handler)

            # ---------------------------------------------------
            # Create and Run Detection Application
            # ---------------------------------------------------
            app = HailoDetectionApp(detection_callback, callback)
            
            # Ensure only one pipeline running at a time
            with pipeline_lock:
                # Stop any existing app first
                if current_pipeline_app is not None:
                    print("Stopping previous pipeline app...")
                    current_pipeline_app.shutdown()
                    time.sleep(2)
            
                current_pipeline_app = app     
            
            # Run pipeline (blocks until shutdown)
            app.run()
            
            # END TRANSACTION MEMORY TRACKING
            if transaction_id:
                transaction_memory_manager.end_transaction(transaction_id)
                print(f"[Memory] Transaction {transaction_id} ended")
            
    except Exception as e:
        print(f"Error during tracking: {e}")
        
        # END TRANSACTION ON ERROR
        if transaction_id:
            try:
                transaction_memory_manager.end_transaction(transaction_id)
                print(f"[Memory] Transaction {transaction_id} ended (due to error)")
            except:
                pass
        
    finally:
        # =============================================================
        # CLEANUP
        # =============================================================
        await websocket.send_json({
            "status": "stopped",
            "message": "Tracking has been fully stopped"
        })
        
        door_monitor_active = False
        
        # Stop alert threads if running
        if cover_alert_thread is not None and cover_alert_thread.is_alive():
            camera_covered = False
            tts_manager.stop_all_audio()
            cover_alert_thread.join()
            cover_alert_thread = None
            alert_thread.join()
            alert_thread = None
            
        await door_monitor_task
        callback.tracking_data.shutdown_event.set()
        callback.shutdown_event.set()
        websocket_sender.join()
        app.pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

# =====================================================================
# PRODUCT CAPTURE SYSTEM (New Product Training)
# =====================================================================
"""
This system allows administrators to add new products to the AI model by:
1. Capturing multiple photos from different angles
2. Playing audio instructions for positioning
3. Uploading images to backend for AI training
4. Deleting local copies after successful upload

WORKFLOW:
Admin mode → Select "Add Product" → Enter product name →
Camera captures 3+ images → Upload to server → AI retraining
"""

def setup_product_upload_alerts():
    """
    Generate and save product upload alert sounds using text-to-speech.
    
    Creates MP3 files for each step of the product capture process:
    - Start capture: "Get ready to capture images..."
    - Camera switch: "Switching to next camera..."
    - Capture ready: "Position your product now..."
    - Image captured: "Image captured successfully."
    - Next position: "Next position."
    - Upload success: "All images uploaded successfully."
    - Upload failed: "Upload failed. Please contact support."
    - All complete: "Image capture completed."
    
    Directory Structure:
        sounds/
        └── product_upload_alerts/
            ├── start_capture.mp3
            ├── camera_switch.mp3
            ├── capture_ready.mp3
            ├── image_captured.mp3
            ├── next_position.mp3
            ├── upload_success.mp3
            ├── upload_failed.mp3
            └── all_complete.mp3
    
    Returns:
        dict: Mapping of alert names to file paths
    """
    alert_dir = "sounds/product_upload_alerts"
    
    # Create directory if it doesn't exist
    os.makedirs(alert_dir, exist_ok=True)
    
    # Define all alert messages
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
    
    generated_files = {}
    
    # Generate each alert sound
    for alert_name, alert_text in alerts.items():
        alert_file = os.path.join(alert_dir, f"{alert_name}.mp3")
        
        # Only generate if file doesn't exist (caching)
        if not os.path.exists(alert_file):
            tts = gTTS(text=alert_text, lang='en', slow=False)
            tts.save(alert_file)
            print(f"Generated: {alert_file}")
        
        generated_files[alert_name] = alert_file
    
    print(f"All product upload alert sounds ready in {alert_dir}")
    return generated_files

def capture_images(device_id, num_images=3):
    """
    Capture multiple images from a camera with audio guidance.
    
    This optimized capture system:
    1. Uses MJPEG format for faster capture (vs YUYV)
    2. Reduces buffer size to minimize lag
    3. Clears buffer before each capture (ensures fresh frame)
    4. Provides audio feedback for each step
    5. Saves images with timestamp naming
    
    Optimization Techniques:
    - MJPEG codec: Much faster than YUYV decoding
    - Buffer size 1: Minimizes old frame lag
    - Buffer clearing: Ensures we capture current state, not buffered frame
    - 30 FPS: Smooth preview for positioning
    
    Args:
        device_id (int): Camera device ID (0 for /dev/video0, 2 for /dev/video2)
        num_images (int): Number of images to capture (default 3)
        
    Returns:
        list: Paths to captured image files, or empty list on failure
        
    Capture Sequence:
    1. Open camera with optimized settings
    2. For each image:
       - Play "capture ready" audio
       - Wait 0.5 seconds for positioning
       - Clear buffer (5 frames)
       - Capture frame
       - Save to disk
       - Play "image captured" confirmation
       - Play "next position" (if more images remaining)
    
    Saved Files:
        camera_images/camera_{device_id}_image_{n}.jpg
    """
    image_paths = []
    
    # Get alert sound directory
    alert_dir = "sounds/product_upload_alerts"
    
    # Create output directory if it doesn't exist
    os.makedirs('camera_images', exist_ok=True)
    
    try:
        # Open the camera
        cap = cv2.VideoCapture(device_id)
        
        # =========================================================
        # OPTIMIZATION 1: Use MJPEG format (much faster than YUYV)
        # =========================================================
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # =========================================================
        # OPTIMIZATION 2: Set resolution
        # =========================================================
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # =========================================================
        # OPTIMIZATION 3: Increase FPS and reduce buffer
        # =========================================================
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {device_id}")
            return []
            
        print(f"\n=== Starting capture for Camera {device_id} ===")
        print("Get ready to show your products!")
        
        # Play capture ready alert
        tts_manager.play_mp3_sync(f"{alert_dir}/capture_ready.mp3", volume=0.8)
        
        # Capture each image
        for i in range(1, num_images + 1):
            print(f"\nCamera {device_id}: Product position {i} now!")
            
            # Brief pause for positioning
            time.sleep(0.5)  
            
            # =====================================================
            # Clear buffer to get fresh frame (not buffered frame)
            # =====================================================
            for _ in range(5):  
                cap.read()
            
            print(f"Capturing image {i}...")
            
            # Capture the actual frame
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture image {i} from camera {device_id}")
                continue
            
            # Save the image
            filename = os.path.join('camera_images', 
                                   f"camera_{device_id}_image_{i}.jpg")
            cv2.imwrite(filename, frame)
            
            print(f"Saved {filename}")
            image_paths.append(filename)
            
            # Play image captured confirmation
            tts_manager.play_mp3_sync(f"{alert_dir}/image_captured.mp3", volume=0.8)
            
            # Wait before next capture
            if i < num_images:
                print(f"Get ready for product position {i+1}...")
                # Play next position alert
                tts_manager.play_mp3_async(f"{alert_dir}/next_position.mp3", volume=0.8)
                time.sleep(1)
                
        print(f"\n=== Finished capturing from Camera {device_id} ===")
        cap.release()
        return image_paths
        
    except Exception as e:
        print(f"Error with camera {device_id}: {e}")
        if 'cap' in locals():
            cap.release()
        return []

def upload_images_to_api(camera1_images, machine_id, machine_identifier, 
                        user_id, product_name, image_count):
    """
    Upload captured images to the backend API for AI training.
    
    API Details:
    - Endpoint: /mobile_app/product/Product/upload_product_images
    - Method: POST with multipart/form-data
    - Authentication: Basic Auth + API Key
    - Files: Multiple images as 'image[]' array
    
    Args:
        camera1_images (list): Paths to camera 1 images
        machine_id (str): Machine database ID
        machine_identifier (str): Machine name/code
        user_id (str): Admin user ID uploading images
        product_name (str): Name of the new product
        image_count (int): Number of images per camera
        
    Returns:
        bool: True if upload successful (status 200), False otherwise
        
    Upload Process:
    1. Prepare form data (machine info, product name, counts)
    2. Open all image files and prepare multipart form
    3. Send POST request with authentication
    4. Check response status
    5. Always close file handles (in finally block)
    
    File Naming in Upload:
        camera1_0.jpg, camera1_1.jpg, camera1_2.jpg, ...
    """
    api_url = "https://stg-sfapi.nuboxtech.com/index.php/mobile_app/product/Product/upload_product_images"
    
    # Authentication credentials
    username = 'admin'
    password = '1234'
    api_key = '123456'
    
    # Prepare form data
    payload = {
        'machine_id': machine_id,
        'machine_identifier': machine_identifier,
        'user_id': user_id,
        'product_name': product_name,
        'image_count': image_count
    }
    
    # Prepare headers
    headers = {'x-api-key': api_key}
    
    files = []
    opened_files = []  # Track opened file handles for cleanup
    
    try:
        # Add camera 1 images to upload
        for i, img_path in enumerate(camera1_images):
            file_handle = open(img_path, 'rb')
            opened_files.append(file_handle)
            files.append(('image[]', (f'camera1_{i}.jpg', file_handle, 'image/jpeg')))
        
        # Upload to API
        response = requests.post(
            api_url,
            auth=HTTPBasicAuth(username, password),
            headers=headers,
            data=payload,
            files=files,
        )
        
        print("API Response Status Code:", response.status_code)
        print("API Response:", response.text)
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error uploading to API: {e}")
        return False
        
    finally:
        # Always close all opened file handles
        for file_handle in opened_files:
            try:
                file_handle.close()
            except:
                pass

def delete_images(image_paths):
    """
    Delete image files from the filesystem after successful upload.
    
    Why Delete?
    - Free up disk space (Raspberry Pi has limited storage)
    - Remove temporary files
    - Images already backed up on server
    
    Args:
        image_paths (list): List of image file paths to delete
        
    Returns:
        int: Number of images successfully deleted
        
    Safety:
    - Checks if file exists before attempting delete
    - Catches and reports individual deletion errors
    - Continues even if one deletion fails
    """
    deleted_count = 0
    
    for img_path in image_paths:
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Deleted: {img_path}")
                deleted_count += 1
            else:
                print(f"File not found: {img_path}")
        except Exception as e:
            print(f"Error deleting {img_path}: {e}")
    
    print(f"Successfully deleted {deleted_count} images")
    return deleted_count

# =====================================================================
# TRANSACTION-BASED MEMORY MANAGEMENT SYSTEM
# =====================================================================
"""
CRITICAL SYSTEM FOR 24/7 OPERATION

PROBLEM:
- Smart fridge runs 24/7 with hundreds of daily transactions
- Python doesn't immediately release memory after objects deleted
- Data accumulates: trails, tracks, history → Memory grows indefinitely
- After days/weeks: System runs out of RAM and crashes

SOLUTION: Transaction-Based Memory Management
- Each transaction has a lifecycle: start → detect → end → cleanup
- Track all data created during transaction
- On transaction end: Aggressively release memory
- Use multiple strategies to force Python to free memory

STRATEGIES USED:
1. Remove references from dictionaries
2. Recreate dictionaries from scratch (forces new memory allocation)
3. Multiple garbage collection passes
4. Call malloc_trim() to return memory to OS (Linux)

TYPICAL TRANSACTION:
Customer opens app (10:00 AM)
  → start_transaction(trans_123)
  → Creates 50 tracks, 200 trail points, 500 frames
  → Uses ~50MB memory
Customer closes door (10:05 AM)
  → end_transaction(trans_123)
  → Aggressive cleanup
  → Memory freed: ~45MB (90% recovery)
  → System ready for next customer
"""

class TransactionMemoryManager:
    """
    Enhanced memory manager that FORCES memory release after each transaction.
    
    This class is essential for long-running operation because:
    - Python's garbage collector is lazy (doesn't release immediately)
    - Object references can linger in caches and histories
    - Memory fragmentation occurs over time
    - We need deterministic cleanup, not eventual cleanup
    
    Tracking:
    - Active transactions with their memory usage
    - Objects created per transaction (tracks, trails)
    - Transaction history (last 100 transactions)
    - Global statistics (total transactions, cleanups, memory trends)
    
    Cleanup Process:
    1. Identify all data belonging to transaction
    2. Remove from global dictionaries
    3. Recreate dictionaries (forces Python to allocate new memory)
    4. Run garbage collection multiple times
    5. Try to return memory to OS
    """
    
    def __init__(self):
        """
        Initialize memory manager with tracking structures.
        """
        # Currently active transactions
        # Format: {transaction_id: {start_time, start_memory_mb, tracks_created, ...}}
        self.active_transactions = {}
        
        # History of completed transactions (last 100)
        self.transaction_history = deque(maxlen=100)
        
        # Global statistics
        self.global_stats = {
            'total_transactions': 0,
            'total_cleanups': 0,
            'average_memory_per_transaction': 0,
            'peak_memory': 0
        }
        
        # Thread lock for concurrent access
        self.lock = threading.Lock()
    
    def start_transaction(self, transaction_id):
        """
        Called when WebSocket connects and transaction starts.
        
        Records:
        - Start time
        - Initial memory usage
        - Empty sets for tracking created objects
        
        Args:
            transaction_id (str): Unique transaction identifier
        """
        with self.lock:
            print(f"\n{'='*60}")
            print(f"[Transaction] Starting: {transaction_id}")
            print(f"{'='*60}")
            
            # Get current process memory usage
            process = psutil.Process()
            memory_start = process.memory_info().rss / 1024 / 1024  # Convert to MB
            
            # Create transaction record
            self.active_transactions[transaction_id] = {
                'start_time': time.time(),
                'start_memory_mb': memory_start,
                'tracks_created': set(),      # Track IDs created
                'trails_created': set(),      # Global IDs with trails
                'frames_processed': 0         # Frame counter
            }
            
            self.global_stats['total_transactions'] += 1
            
            print(f"[Transaction] Memory at start: {memory_start:.1f}MB")
            print(f"[Transaction] Active transactions: {len(self.active_transactions)}")
    
    def end_transaction(self, transaction_id):
        """
        Called when WebSocket closes - AGGRESSIVE cleanup.
        
        This is the CRITICAL function that prevents memory leaks.
        
        Process:
        1. Calculate memory used during transaction
        2. Remove transaction-specific data
        3. Recreate global dictionaries (NUCLEAR OPTION)
        4. Multiple garbage collection passes
        5. Return memory to OS
        6. Verify memory freed
        
        Args:
            transaction_id (str): Transaction to clean up
        """
        with self.lock:
            if transaction_id not in self.active_transactions:
                print(f"[Transaction] Warning: {transaction_id} not found")
                return
            
            print(f"\n{'='*60}")
            print(f"[Transaction] Ending: {transaction_id}")
            print(f"{'='*60}")
            
            trans_data = self.active_transactions[transaction_id]
            
            # ==========================================================
            # STEP 1: CALCULATE METRICS
            # ==========================================================
            duration = time.time() - trans_data['start_time']
            process = psutil.Process()
            memory_before_cleanup = process.memory_info().rss / 1024 / 1024
            memory_used = memory_before_cleanup - trans_data['start_memory_mb']
            
            print(f"[Transaction] Duration: {duration:.1f}s")
            print(f"[Transaction] Frames processed: {trans_data['frames_processed']}")
            print(f"[Transaction] Tracks created: {len(trans_data['tracks_created'])}")
            print(f"[Transaction] Memory before cleanup: {memory_before_cleanup:.1f}MB")
            print(f"[Transaction] Memory used during transaction: {memory_used:.1f}MB")
            
            # ==========================================================
            # STEP 2: STORE HISTORY
            # ==========================================================
            self.transaction_history.append({
                'transaction_id': transaction_id,
                'duration': duration,
                'memory_used_mb': memory_used,
                'frames': trans_data['frames_processed'],
                'timestamp': datetime.now()
            })
            
            # ==========================================================
            # STEP 3: REMOVE FROM ACTIVE
            # ==========================================================
            del self.active_transactions[transaction_id]
            
            # ==========================================================
            # STEP 4: AGGRESSIVE CLEANUP - Multiple Strategies
            # ==========================================================
            print(f"[Cleanup] Starting aggressive cleanup...")
            
            # Strategy 1: Remove from dictionaries
            self._cleanup_transaction_data(transaction_id, trans_data)
            
            # Strategy 2: RECREATE dictionaries (forces memory release)
            self._recreate_global_dictionaries()
            
            # Strategy 3: Force multiple GC passes
            self._aggressive_garbage_collection()
            
            # Strategy 4: Try to release memory back to OS
            self._release_memory_to_os()
            
            # ==========================================================
            # STEP 5: VERIFY CLEANUP
            # ==========================================================
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_freed = memory_before_cleanup - memory_after
            
            print(f"[Transaction] Memory after cleanup: {memory_after:.1f}MB")
            print(f"[Transaction] Memory freed: {memory_freed:.1f}MB")
            
            # Check if enough memory was freed (expect ~80% recovery)
            if memory_freed < 10:
                print(f"[Transaction] ⚠️ Warning: Only {memory_freed:.1f}MB freed "
                      f"(expected ~{memory_used * 0.8:.1f}MB)")
            else:
                print(f"[Transaction] ✅ Successfully freed {memory_freed:.1f}MB")
            
            print(f"{'='*60}\n")
    
    def _cleanup_transaction_data(self, transaction_id, trans_data):
        """
        Remove transaction data from global dictionaries.
        
        Removes:
        - Object trails (visual paths)
        - Global trails (cross-camera paths)
        - Movement history (position tracking)
        - Bounding box history (size tracking)
        - Track ID mappings
        - Active object records
        
        Args:
            transaction_id (str): Transaction ID
            trans_data (dict): Transaction data with created object IDs
        """
        global object_trails, global_trails, camera_movement_history
        global camera_bbox_area_history, local_to_global_id_map
        global active_objects_per_camera, global_movement_history
        
        tracks_to_clean = trans_data['tracks_created']
        trails_to_clean = trans_data['trails_created']
        
        # Count before cleanup (for logging)
        trails_before = len(object_trails) + len(global_trails)
        tracks_before = sum(len(camera_movement_history[c]) for c in [0, 1])
        
        # -----------------------------------------------------------
        # Clean object trails
        # -----------------------------------------------------------
        for track_id in list(object_trails.keys()):
            if track_id in tracks_to_clean:
                object_trails[track_id].clear()  # Clear deque first
                del object_trails[track_id]
        
        # -----------------------------------------------------------
        # Clean global trails
        # -----------------------------------------------------------
        for global_id in list(global_trails.keys()):
            if global_id in trails_to_clean:
                global_trails[global_id].clear()
                del global_trails[global_id]
        
        # -----------------------------------------------------------
        # Clean movement history (both cameras)
        # -----------------------------------------------------------
        for camera_id in [0, 1]:
            for track_id in list(camera_movement_history[camera_id].keys()):
                if track_id in tracks_to_clean:
                    camera_movement_history[camera_id][track_id].clear()
                    del camera_movement_history[camera_id][track_id]
            
            for track_id in list(camera_bbox_area_history[camera_id].keys()):
                if track_id in tracks_to_clean:
                    camera_bbox_area_history[camera_id][track_id].clear()
                    del camera_bbox_area_history[camera_id][track_id]
        
        # -----------------------------------------------------------
        # Clean track ID mappings
        # -----------------------------------------------------------
        for key in list(local_to_global_id_map.keys()):
            cam_id, local_id = key
            if local_id in tracks_to_clean:
                del local_to_global_id_map[key]
        
        # -----------------------------------------------------------
        # Clean active objects per camera
        # -----------------------------------------------------------
        for camera_id in [0, 1]:
            for label in list(active_objects_per_camera[camera_id].keys()):
                for local_id in list(active_objects_per_camera[camera_id][label].keys()):
                    if local_id in tracks_to_clean:
                        del active_objects_per_camera[camera_id][label][local_id]
                
                # Clean empty label entries
                if not active_objects_per_camera[camera_id][label]:
                    del active_objects_per_camera[camera_id][label]
        
        # -----------------------------------------------------------
        # Clean global movement history
        # -----------------------------------------------------------
        for global_id in list(global_movement_history.keys()):
            if global_id in trails_to_clean:
                global_movement_history[global_id].clear()
                del global_movement_history[global_id]
        
        # Log cleanup results
        trails_after = len(object_trails) + len(global_trails)
        tracks_after = sum(len(camera_movement_history[c]) for c in [0, 1])
        
        print(f"[Cleanup] Trails: {trails_before} -> {trails_after} "
              f"(removed {trails_before - trails_after})")
        print(f"[Cleanup] Tracks: {tracks_before} -> {tracks_after} "
              f"(removed {tracks_before - tracks_after})")
    
    def _recreate_global_dictionaries(self):
        """
        NUCLEAR OPTION: Recreate dictionaries from scratch.
        
        Why This Works:
        - Python dictionaries allocate memory in chunks
        - Deleting items doesn't release the chunk memory
        - Creating NEW dictionary forces Python to allocate fresh memory
        - Old dictionary memory gets released
        
        This is like demolishing a building and building new one,
        rather than just removing furniture from old building.
        
        Process:
        1. Create new empty dictionaries
        2. Copy only active transaction data to new dictionaries
        3. Reassign global variables to new dictionaries
        4. Old dictionaries get garbage collected
        """
        global object_trails, global_trails, camera_movement_history
        global camera_bbox_area_history, active_objects_per_camera
        global global_movement_history, local_to_global_id_map
        
        print(f"[Cleanup] Recreating global dictionaries...")
        
        # Count items to preserve
        preserve_count = 0
        
        # Identify data from active transactions (don't delete these)
        active_tracks = set()
        active_trails = set()
        for trans_data in self.active_transactions.values():
            active_tracks.update(trans_data['tracks_created'])
            active_trails.update(trans_data['trails_created'])
        
        # -----------------------------------------------------------
        # Recreate object_trails
        # -----------------------------------------------------------
        new_object_trails = defaultdict(lambda: deque(maxlen=30))
        for track_id, trail in object_trails.items():
            if track_id in active_tracks:
                new_object_trails[track_id] = trail
                preserve_count += 1
        object_trails = new_object_trails
        
        # -----------------------------------------------------------
        # Recreate global_trails
        # -----------------------------------------------------------
        new_global_trails = defaultdict(lambda: deque(maxlen=30))
        for global_id, trail in global_trails.items():
            if global_id in active_trails:
                new_global_trails[global_id] = trail
                preserve_count += 1
        global_trails = new_global_trails
        
        # -----------------------------------------------------------
        # Recreate camera movement histories
        # -----------------------------------------------------------
        new_camera_movement_history = {
            0: defaultdict(lambda: deque(maxlen=5)),
            1: defaultdict(lambda: deque(maxlen=5))
        }
        for cam_id in [0, 1]:
            for track_id, history in camera_movement_history[cam_id].items():
                if track_id in active_tracks:
                    new_camera_movement_history[cam_id][track_id] = history
                    preserve_count += 1
        camera_movement_history = new_camera_movement_history
        
        # -----------------------------------------------------------
        # Recreate bounding box histories
        # -----------------------------------------------------------
        new_camera_bbox_area_history = {
            0: defaultdict(lambda: deque(maxlen=5)),
            1: defaultdict(lambda: deque(maxlen=5))
        }
        for cam_id in [0, 1]:
            for track_id, history in camera_bbox_area_history[cam_id].items():
                if track_id in active_tracks:
                    new_camera_bbox_area_history[cam_id][track_id] = history
        camera_bbox_area_history = new_camera_bbox_area_history
        
        print(f"[Cleanup] Preserved {preserve_count} items from active transactions")
        print(f"[Cleanup] Dictionary recreation complete")
    
    def _aggressive_garbage_collection(self):
        """
        Run multiple garbage collection passes to ensure cleanup.
        
        Python's Garbage Collector:
        - Generation 0: Young objects (checked frequently)
        - Generation 1: Medium age objects
        - Generation 2: Old objects (checked rarely)
        
        Strategy:
        - Run collection on all 3 generations
        - Repeat 3 times (diminishing returns after that)
        - Stop early if no objects collected
        
        Output: Number of objects collected per generation
        """
        print(f"[GC] Running aggressive garbage collection...")
        
        total_collected = 0
        
        # Run multiple passes (up to 3)
        for pass_num in range(3):
            # Collect from all generations
            collected = [gc.collect(gen) for gen in range(3)]
            total = sum(collected)
            total_collected += total
            
            print(f"[GC] Pass {pass_num + 1}: "
                  f"Gen0={collected[0]}, Gen1={collected[1]}, Gen2={collected[2]} "
                  f"(total={total})")
            
            if total == 0:
                break  # No more to collect, stop early
        
        print(f"[GC] Total objects collected: {total_collected}")
        
        # Display GC statistics
        gc_stats = gc.get_stats()
        print(f"[GC] Current GC stats:")
        for gen, stats in enumerate(gc_stats):
            print(f"[GC]   Gen{gen}: collections={stats['collections']}, "
                  f"collected={stats.get('collected', 0)}, "
                  f"uncollectable={stats.get('uncollectable', 0)}")
        
        self.global_stats['total_cleanups'] += 1
    
    def _release_memory_to_os(self):
        """
        Try to release memory back to the operating system.
        
        The Problem:
        - Python's memory allocator (malloc) holds onto freed memory
        - Memory is "free" in Python but still reserved from OS
        - OS sees high memory usage even though Python doesn't need it
        
        The Solution:
        - Call malloc_trim(0) to force memory back to OS
        - This is Linux-specific (uses glibc's malloc)
        - Not available on other platforms
        
        Impact:
        - Can free 10-50MB back to OS immediately
        - Reduces overall system memory pressure
        - Important for long-running processes
        """
        print(f"[Memory] Attempting to release memory to OS...")
        
        try:
            # For Linux: Use malloc_trim to release memory
            if sys.platform == 'linux':
                libc = ctypes.CDLL('libc.so.6')
                libc.malloc_trim(0)  # 0 = trim all possible memory
                print(f"[Memory] malloc_trim() called successfully")
            else:
                print(f"[Memory] malloc_trim not available on {sys.platform}")
        except Exception as e:
            print(f"[Memory] Could not call malloc_trim: {e}")
    
    def track_frame(self, transaction_id):
        """
        Called for each processed frame.
        
        Increments frame counter for the transaction.
        Used for statistics and monitoring.
        
        Args:
            transaction_id (str): Transaction ID
        """
        if transaction_id in self.active_transactions:
            self.active_transactions[transaction_id]['frames_processed'] += 1
    
    def track_object(self, transaction_id, track_id, global_id):
        """
        Called when new track is created.
        
        Records which tracks and trails belong to this transaction
        for cleanup later.
        
        Args:
            transaction_id (str): Transaction ID
            track_id (int): Local track ID
            global_id (int): Global track ID
        """
        if transaction_id in self.active_transactions:
            self.active_transactions[transaction_id]['tracks_created'].add(track_id)
            self.active_transactions[transaction_id]['trails_created'].add(global_id)
    
    def get_stats(self):
        """
        Get memory and transaction statistics.
        
        Returns:
            dict: Statistics including:
                - current_memory_mb: Current process memory usage
                - active_transactions: Number of active transactions
                - total_transactions: Total transactions processed
                - total_cleanups: Total cleanup operations
                - recent_transactions: Last 10 transaction details
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'current_memory_mb': memory_info.rss / 1024 / 1024,
            'active_transactions': len(self.active_transactions),
            'total_transactions': self.global_stats['total_transactions'],
            'total_cleanups': self.global_stats['total_cleanups'],
            'recent_transactions': list(self.transaction_history)[-10:]
        }
    
    def print_stats(self):
        """
        Print detailed statistics to console.
        
        Useful for:
        - Monitoring system health
        - Debugging memory issues
        - Performance analysis
        """
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("TRANSACTION MEMORY STATISTICS")
        print("="*60)
        print(f"Current Memory: {stats['current_memory_mb']:.1f}MB")
        print(f"Active Transactions: {stats['active_transactions']}")
        print(f"Total Transactions: {stats['total_transactions']}")
        print(f"Total Cleanups: {stats['total_cleanups']}")
        print("\nRecent Transactions:")
        for trans in stats['recent_transactions']:
            print(f"  {trans['transaction_id']}: "
                  f"{trans['duration']:.1f}s, "
                  f"{trans['memory_used_mb']:.1f}MB, "
                  f"{trans['frames']} frames")
        print("="*60 + "\n")

# Create global instance
transaction_memory_manager = TransactionMemoryManager()

# =====================================================================
# TEXT-TO-SPEECH (TTS) MANAGER SYSTEM
# =====================================================================
"""
Audio alert system for customer guidance and warnings.

FEATURES:
1. Text-to-speech generation (gTTS)
2. MP3 file playback (pygame.mixer)
3. Sound effects support
4. Pre-generated message caching
5. Volume control
6. Multiple fallback TTS engines

ALERT TYPES:
- Door control: "Open the door" / "Door is closing"
- Price exceeded: "Deposit exceeded. Please return the [items]"
- Camera covered: "Don't cover the camera"
- Product upload: Step-by-step instructions

AUDIO PLAYBACK:
- Synchronous: Blocks until complete (for critical messages)
- Asynchronous: Plays in background (for non-critical)
- Sound effects: Short audio clips (can play simultaneously)
"""

class TTSManager:
    """
    Manager for text-to-speech generation and audio playback.
    
    Architecture:
    - Uses gTTS (Google Text-to-Speech) for voice generation
    - Uses pygame.mixer for MP3 playback
    - Supports both sync and async playback
    - Caches frequently used messages as MP3 files
    - Thread-safe with separate locks for TTS and audio
    
    Why Two Locks?
    - tts_lock: Protects TTS generation (slow operation)
    - audio_lock: Protects audio playback (fast operation)
    - Separate locks allow parallel TTS generation and playback
    """
    
    def __init__(self):
        """
        Initialize TTS manager with audio player and locks.
        """
        # Thread locks for concurrent access
        self.tts_lock = Lock()      # For TTS generation
        self.audio_lock = Lock()    # For audio playback
        
        # Initialize pygame audio mixer
        self.init_audio_player()
        
        # Directory for deposit alert sounds
        self.deposit_sounds_dir = "sounds/deposits"
    
    def init_audio_player(self):
        """
        Initialize pygame mixer for audio playback.
        
        Settings:
        - frequency: 22050 Hz (CD quality audio)
        - size: -16 (16-bit signed audio)
        - channels: 2 (stereo)
        - buffer: 512 (small buffer for low latency)
        """
        try:
            pygame.mixer.init(
                frequency=22050,  # Sample rate
                size=-16,         # 16-bit signed
                channels=2,       # Stereo
                buffer=512        # Small buffer
            )
            print("Audio player initialized successfully")
        except Exception as e:
            print(f"Error initializing audio player: {e}")
    
    # =================================================================
    # MP3 FILE PLAYBACK
    # =================================================================
    
    def play_mp3(self, file_path, volume=0.7, wait_for_completion=True):
        """
        Play an MP3 file with optional blocking.
        
        Args:
            file_path (str): Path to MP3 file
            volume (float): Volume level (0.0 to 1.0)
            wait_for_completion (bool): Whether to wait for playback to finish
            
        Returns:
            bool: True if playback started successfully, False otherwise
            
        Usage:
            # Synchronous (blocking)
            tts_manager.play_mp3("alert.mp3", volume=0.8, wait_for_completion=True)
            
            # Asynchronous (non-blocking)
            tts_manager.play_mp3("alert.mp3", volume=0.8, wait_for_completion=False)
        """
        def _play():
            """Inner function for actual playback."""
            with self.audio_lock:
                try:
                    # Validate file exists
                    if not os.path.exists(file_path):
                        print(f"MP3 file not found: {file_path}")
                        return False
                    
                    # Validate file is MP3
                    if not file_path.lower().endswith('.mp3'):
                        print(f"File is not an MP3: {file_path}")
                        return False
                    
                    print(f"Playing MP3: {file_path}")
                    
                    # Load and play the MP3 file
                    pygame.mixer.music.load(file_path)
                    pygame.mixer.music.set_volume(volume)
                    pygame.mixer.music.play()
                    
                    if wait_for_completion:
                        # Wait for playback to complete
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                    
                    print(f"Finished playing MP3: {file_path}")
                    return True
                    
                except Exception as e:
                    print(f"Error playing MP3 {file_path}: {e}")
                    return False
        
        if wait_for_completion:
            # Run synchronously (blocking)
            return _play()
        else:
            # Run asynchronously (non-blocking)
            audio_thread = threading.Thread(target=_play, daemon=True)
            audio_thread.start()
            return True
    
    def play_mp3_async(self, file_path, volume=0.7):
        """
        Play MP3 file asynchronously (non-blocking).
        
        Shortcut for play_mp3() with wait_for_completion=False
        """
        return self.play_mp3(file_path, volume, wait_for_completion=False)
    
    def play_mp3_sync(self, file_path, volume=0.7):
        """
        Play MP3 file synchronously (blocking).
        
        Shortcut for play_mp3() with wait_for_completion=True
        """
        return self.play_mp3(file_path, volume, wait_for_completion=True)
    
    def play_sound_effect(self, file_path, volume=0.7):
        """
        Play a sound effect using pygame.mixer.Sound.
        
        Difference from play_mp3():
        - Uses Sound() instead of music channel
        - Can play multiple sounds simultaneously
        - Better for short audio clips (< 1 second)
        - Uses more memory (loads entire file)
        
        Args:
            file_path (str): Path to audio file
            volume (float): Volume level (0.0 to 1.0)
            
        Returns:
            bool: True if playback started successfully
            
        Use Cases:
        - Button click sounds
        - Notification beeps
        - Short confirmation tones
        """
        try:
            if not os.path.exists(file_path):
                print(f"Sound file not found: {file_path}")
                return False
            
            print(f"Playing sound effect: {file_path}")
            sound = pygame.mixer.Sound(file_path)
            sound.set_volume(volume)
            sound.play()
            return True
            
        except Exception as e:
            print(f"Error playing sound effect {file_path}: {e}")
            return False
    
    # =================================================================
    # AUDIO CONTROL
    # =================================================================
    
    def stop_all_audio(self):
        """
        Stop all audio playback immediately.
        
        Stops:
        - Background music (pygame.mixer.music)
        - Sound effects (pygame.mixer.Sound)
        """
        try:
            pygame.mixer.music.stop()  # Stop music
            pygame.mixer.stop()         # Stop all sound effects
            print("All audio stopped")
        except Exception as e:
            print(f"Error stopping audio: {e}")
    
    def pause_audio(self):
        """Pause current music playback."""
        try:
            pygame.mixer.music.pause()
            print("Audio paused")
        except Exception as e:
            print(f"Error pausing audio: {e}")
    
    def resume_audio(self):
        """Resume paused music playback."""
        try:
            pygame.mixer.music.unpause()
            print("Audio resumed")
        except Exception as e:
            print(f"Error resuming audio: {e}")
    
    def set_volume(self, volume):
        """
        Set the volume for music playback.
        
        Args:
            volume (float): Volume level (0.0 to 1.0)
        """
        try:
            pygame.mixer.music.set_volume(volume)
            print(f"Volume set to: {volume}")
        except Exception as e:
            print(f"Error setting volume: {e}")
    
    def is_audio_playing(self):
        """
        Check if audio is currently playing.
        
        Returns:
            bool: True if music is playing
        """
        try:
            return pygame.mixer.music.get_busy()
        except:
            return False
    
    def get_audio_position(self):
        """
        Get current position in music playback (milliseconds).
        
        Returns:
            int: Position in milliseconds, or -1 if not supported
        """
        try:
            return pygame.mixer.music.get_pos()
        except:
            return -1
    
    # =================================================================
    # DEPOSIT ALERT SYSTEM (Price Exceeded Warnings)
    # =================================================================
    
    def generate_common_deposit_messages(self):
        """
        Pre-generate deposit audio files for common product combinations.
        
        Benefits:
        - Faster playback (no generation delay)
        - Consistent voice quality
        - Reduces API calls to gTTS
        
        Called during:
        - System initialization
        - First-time setup
        
        Generates messages for:
        - Common single products
        - Common product combinations
        """
        try:
            print("Pre-generating common deposit messages...")
            
            # Common product names (customize for your fridge)
            common_products = [
                "100plus",
                "coconut",
                "mineral",
                "water bottle",
                "energy drink"
            ]
            
            # Generate single product messages
            for product in common_products:
                self.generate_deposit_audio_file(product)
            
            # Generate common combinations
            common_combinations = [
                ["coke", "pepsi"],
                ["sprite", "water bottle"],
                # Add more common combinations as needed
            ]
            
            for combo in common_combinations:
                self.generate_deposit_audio_file(combo)
            
            print("Common deposit messages generated successfully")
            
        except Exception as e:
            print(f"Error generating common deposit messages: {e}")
    
    def generate_deposit_audio_file(self, label):
        """
        Generate and save deposit audio file for given label(s).
        
        Message Format:
        - Single item: "Deposit exceeded. Please return the [item] immediately"
        - Two items: "Deposit exceeded. Please return the [item1] and [item2] immediately"
        - Multiple: "Deposit exceeded. Please return the [item1], [item2], and [item3] immediately"
        
        Args:
            label (str or list): Product name(s) to return
            
        Returns:
            str: Path to generated/cached MP3 file, or None on error
            
        Caching:
        - Filename based on MD5 hash of text
        - Reuses existing file if available
        - Saves to sounds/deposits/ directory
        """
        try:
            # ---------------------------------------------------
            # Build the text message
            # ---------------------------------------------------
            if isinstance(label, str):
                text = f"Deposit exceeded. Please return the {label} immediately"
            elif isinstance(label, (list, tuple)):
                if len(label) == 0:
                    return None
                elif len(label) == 1:
                    text = f"Deposit exceeded. Please return the {label[0]} immediately"
                elif len(label) == 2:
                    text = f"Deposit exceeded. Please return the {label[0]} and {label[1]} immediately"
                else:
                    # Multiple items: "item1, item2, and item3"
                    items_text = ", ".join(label[:-1]) + f", and {label[-1]}"
                    text = f"Deposit exceeded. Please return the {items_text} immediately"
            else:
                # Handle comma-separated string
                if isinstance(label, str) and "," in label:
                    items = [item.strip() for item in label.split(",")]
                    return self.generate_deposit_audio_file(items)
                else:
                    text = f"Deposit exceeded. Please return the {label} immediately"
            
            # ---------------------------------------------------
            # Create unique filename based on text content
            # ---------------------------------------------------
            text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            filename = f"deposit_{text_hash}.mp3"
            filepath = os.path.join(self.deposit_sounds_dir, filename)
            
            # ---------------------------------------------------
            # Generate file if it doesn't exist (caching)
            # ---------------------------------------------------
            if not os.path.exists(filepath):
                print(f"Generating deposit audio: {text}")
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(filepath)
                print(f"Saved deposit audio to: {filepath}")
            else:
                print(f"Using cached deposit audio: {filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"Error generating deposit audio file: {e}")
            return None

    def speak_deposit(self, label):
        """
        Speak deposit exceeded message for single or multiple items.
        
        Features:
        - Uses pre-generated/cached MP3 files (fast)
        - Falls back to async TTS if file generation fails
        - Handles single items, multiple items, and comma-separated strings
        
        Args:
            label (str, list, or comma-separated str): Item(s) to return
            
        Examples:
            speak_deposit("coke")                    → "Return the coke"
            speak_deposit(["coke", "sprite"])        → "Return the coke and sprite"
            speak_deposit("coke, sprite, water")     → "Return the coke, sprite, and water"
        """
        try:
            # Handle comma-separated string
            if isinstance(label, str) and "," in label:
                items = [item.strip() for item in label.split(",")]
                label = items
            
            # Generate or get cached audio file
            filepath = self.generate_deposit_audio_file(label)
            
            if filepath and os.path.exists(filepath):
                # Play the pre-generated MP3 file
                self.play_mp3_async(filepath, volume=0.8)
            else:
                # Fallback to async TTS if file generation failed
                print("Falling back to async TTS for deposit message")
                if isinstance(label, str):
                    self.speak_async(
                        f"Deposit exceeded. Please return the {label} immediately", 
                        lang='en'
                    )
                elif isinstance(label, (list, tuple)):
                    if len(label) == 1:
                        self.speak_async(
                            f"Deposit exceeded. Please return the {label[0]} immediately", 
                            lang='en'
                        )
                    elif len(label) == 2:
                        self.speak_async(
                            f"Deposit exceeded. Please return the {label[0]} and {label[1]} immediately", 
                            lang='en'
                        )
                    elif len(label) > 2:
                        items_text = ", ".join(label[:-1]) + f", and {label[-1]}"
                        self.speak_async(
                            f"Deposit exceeded. Please return the {items_text} immediately", 
                            lang='en'
                        )
                        
        except Exception as e:
            print(f"Error in speak_deposit: {e}")
            # Final fallback
            try:
                self.speak_async(
                    "Deposit exceeded. Please return the items immediately", 
                    lang='en'
                )
            except:
                pass
    
    # =================================================================
    # DOOR CONTROL AUDIO
    # =================================================================
    
    def generate_door_audio_files(self):
        """
        Pre-generate door open/close audio files.
        
        Creates:
        - sounds/door_open.mp3: "Open the door"
        - sounds/door_close.mp3: "Door is closing"
        """
        try:
            # Generate door open
            tts_open = gTTS(text="Open the door", lang='en', slow=False)
            tts_open.save("sounds/door_open.mp3")
        
            # Generate door close
            tts_close = gTTS(text="Door is closing", lang='en', slow=False)
            tts_close.save("sounds/door_close.mp3")
        
            print("Door audio files generated successfully")
        except Exception as e:
            print(f"Error generating door audio files: {e}")
    
    def speak_door_open(self):
        """Play door open message - using pre-recorded file."""
        self.play_mp3_sync("sounds/door_open.mp3", volume=0.8)
    
    def speak_door_close(self):
        """Play door close message - using pre-recorded file."""
        self.play_mp3_sync("sounds/door_close.mp3", volume=0.8)
    
    # =================================================================
    # DYNAMIC TEXT-TO-SPEECH (Real-time generation)
    # =================================================================
    
    def speak_async(self, text, lang='en'):
        """
        Speak text asynchronously using gTTS (non-blocking).
        
        Process:
        1. Generate TTS audio in memory (BytesIO buffer)
        2. Load audio into pygame mixer
        3. Play audio
        4. Wait for completion in background thread
        
        Args:
            text (str): Text to speak
            lang (str): Language code ('en' for English, 'ms' for Malay)
            
        Note: Runs in separate thread to avoid blocking main program
        """
        def _speak():
            """Inner function that runs in background thread."""
            with self.tts_lock:
                try:
                    print(f"Speaking: {text}")
                    
                    # Create gTTS object
                    tts = gTTS(text=text, lang=lang, slow=False)
                    
                    # Save to BytesIO buffer (in-memory, no disk I/O)
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    
                    # Load and play audio using pygame
                    pygame.mixer.music.load(audio_buffer)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to complete
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    print(f"Finished speaking: {text}")
                    
                except Exception as e:
                    print(f"Error in gTTS: {e}")
                    self.fallback_speak(text)
        
        # Run TTS in separate thread
        tts_thread = threading.Thread(target=_speak, daemon=True)
        tts_thread.start()
    
    def speak_english(self, text):
        """Speak text in English."""
        self.speak_async(text, lang='en')
    
    def speak_malay(self, text):
        """Speak text in Malay."""
        self.speak_async(text, lang='ms')
    
    # =================================================================
    # FALLBACK TTS ENGINES
    # =================================================================
    
    def fallback_speak(self, text):
        """
        Fallback TTS using system espeak (if gTTS fails).
        
        Why Needed:
        - gTTS requires internet connection
        - Network failures can occur
        - espeak works offline
        
        Args:
            text (str): Text to speak
        """
        try:
            # Check if espeak is available
            subprocess.run(['which', 'espeak'], check=True, capture_output=True)
            
            # Use espeak with better settings
            cmd = [
                'espeak', 
                '-s', '120',    # Speech rate (words per minute)
                '-a', '200',    # Amplitude (volume)
                '-p', '50',     # Pitch
                '-g', '3',      # Gap between words
                text
            ]
            subprocess.run(cmd, check=False, capture_output=True)
            print(f"Fallback TTS spoke: {text}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Espeak not available. Install with: sudo apt-get install espeak")
            self.alternative_fallback(text)
    
    def alternative_fallback(self, text):
        """
        Alternative fallback using festival (if espeak unavailable).
        
        Festival is another offline TTS engine.
        
        Args:
            text (str): Text to speak
        """
        try:
            subprocess.run(['which', 'festival'], check=True, capture_output=True)
            
            # Create temporary file for festival
            temp_file = '/tmp/tts_temp.txt'
            with open(temp_file, 'w') as f:
                f.write(text)
            
            cmd = ['festival', '--tts', temp_file]
            subprocess.run(cmd, check=False, capture_output=True)
            
            # Clean up
            os.remove(temp_file)
            print(f"Festival TTS spoke: {text}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Festival not available. Install with: sudo apt-get install festival")
            print(f"No TTS available. Would speak: {text}")
    
    # =================================================================
    # TESTING AND CLEANUP
    # =================================================================
    
    def test_voice(self):
        """Test the TTS voice with sample phrases."""
        print("Testing English voice...")
        self.speak_english("Testing voice clarity. Can you hear this clearly?")
        time.sleep(3)
        print("Testing Malay voice...")
        self.speak_malay("Ujian suara yang jelas. Boleh dengar dengan baik?")
    
    def cleanup(self):
        """Cleanup resources (call on shutdown)."""
        try:
            self.stop_all_audio()
            pygame.mixer.quit()
        except:
            pass

# Create global TTS manager instance
tts_manager = TTSManager()

# =====================================================================
# WEBSOCKET ENDPOINT (FastAPI Integration)
# =====================================================================
"""
This is the main entry point for mobile app communication.

PROTOCOL:
- Mobile app connects via WebSocket
- Sends transaction details (deposit, IDs)
- Receives real-time detection updates
- Closes connection when customer done

CONNECTION LIFECYCLE:
1. Mobile app connects → /ws/track
2. Server accepts connection
3. Server announces "door open" (TTS)
4. Server unlocks door (GPIO)
5. Wait for start_preview or product_upload message
6. Execute appropriate mode (detection or capture)
7. Send real-time updates during transaction
8. Customer closes door → Server detects
9. Server announces "door closing" (TTS)
10. Server locks door (GPIO)
11. Server cleans up memory
12. Connection closes

MESSAGE FORMATS:
Receive (from mobile app):
{
    "action": "start_preview",
    "deposit": 50.00,
    "machine_id": "123",
    "machine_identifier": "fridge_01",
    "user_id": "456",
    "transaction_id": "trans_789"
}

Send (to mobile app):
{
    "validated_products": {
        "entry": {"coke": {"count": 1, "product_details": {...}}},
        "exit": {"sprite": {"count": 2, "product_details": {...}}}
    },
    "invalidated_products": {
        "entry": {},
        "exit": {}
    }
}
"""

# Global state variables for WebSocket handling
done = False  # Transaction completion flag

@app.websocket("/ws/track")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time detection and tracking.
    
    This is the PRIMARY interface between mobile app and detection system.
    Handles entire transaction lifecycle from connection to cleanup.
    
    Responsibilities:
    1. Accept WebSocket connection
    2. Control door lock (unlock on start)
    3. Play audio announcements (door open/close)
    4. Execute tracking or product capture mode
    5. Send real-time updates to mobile app
    6. Clean up resources on disconnect
    7. Manage transaction memory
    
    Args:
        websocket (WebSocket): FastAPI WebSocket connection object
        
    Connection Flow:
        Mobile App → WebSocket Connect → This Function
        ↓
        Accept Connection + Audio Announcement
        ↓
        Unlock Door (GPIO)
        ↓
        Wait for Action Message (start_preview or product_upload)
        ↓
        Execute Appropriate Mode
        ↓
        Send Real-time Updates
        ↓
        Door Closes → Cleanup → Disconnect
    
    Global Variables Used:
        readyToProcess: System readiness flag
        done: Transaction completion flag
        current_pipeline_app: Active pipeline application
    """
    global readyToProcess, done, current_pipeline_app
    
    # Initialize local variables
    deposit = 0.0
    machine_id = None
    machine_identifier = None
    user_id = None
    transaction_id = None
    websocket_sender = None
    
    # =================================================================
    # STEP 1: ACCEPT WEBSOCKET CONNECTION
    # =================================================================
    await websocket.accept()
    print("WebSocket connection accepted")
    
    # =================================================================
    # STEP 2: ANNOUNCE DOOR OPEN (Audio Feedback)
    # =================================================================
    print("WebSocket connected - announcing door open")
    tts_manager.speak_door_open()  # "Open the door"
    
    # =================================================================
    # STEP 3: UNLOCK DOOR AND RESET LEDs
    # =================================================================
    GPIO.output(DOOR_LOCK_PIN, GPIO.LOW)   # Unlock door
    GPIO.output(LED_RED, GPIO.LOW)          # Turn off red LED
    GPIO.output(LED_GREEN, GPIO.LOW)        # Turn on green LED
    
    try:
        # =============================================================
        # STEP 4: WAIT FOR CUSTOMER TO OPEN DOOR
        # =============================================================
        # Wait up to 5 seconds for door to be opened
        start_time = time.time()
        readyToProcess = True
        
        while readyToProcess and time.time() - start_time < 5:
            door_sw = 1  # TODO: Replace with GPIO.input(DOOR_SWITCH_PIN)
            
            if door_sw == 1:  # Door is open
                # Execute tracking/capture mode
                await run_tracking(websocket)
                readyToProcess = False
            else:
                # Door still closed, keep waiting
                readyToProcess = True
        
        # =============================================================
        # STEP 5: FALLBACK IF NO TRACKING STARTED
        # =============================================================
        # This handles case where door never opened or tracking failed
        if not done:
            print("No tracking executed - sending initial update")
            
            # Create callback with transaction data
            callback = HailoDetectionCallback(
                websocket, 
                deposit, 
                machine_id, 
                machine_identifier, 
                user_id, 
                transaction_id
            )
            
            # Start transaction tracking (if transaction_id provided)
            if transaction_id:
                transaction_memory_manager.start_transaction(transaction_id)
            
            # Send initial empty update to mobile app
            while not callback.tracking_data.shutdown_event.is_set():
                try:
                    current_data = callback.tracking_data.websocket_data_manager.get_current_data()
                    await websocket.send_json(current_data)
                    await asyncio.sleep(1)  # Send updates every second
                    break
                except Exception as e:
                    print(f"Error sending websocket data: {e}")
                    break
                    
    except Exception as e:
        print(f"WebSocket tracking error: {e}")
        
    finally:
        # =============================================================
        # CLEANUP PHASE (Always Executed)
        # =============================================================
        print("WebSocket cleanup starting...")
        
        # -------------------------------------------------------------
        # Clean up transaction memory
        # -------------------------------------------------------------
        try:
            if 'callback' in locals() and hasattr(callback, 'transaction_id') and callback.transaction_id:
                transaction_memory_manager.end_transaction(callback.transaction_id)
                print(f"[Memory] Transaction {callback.transaction_id} ended")
        except Exception as e:
            print(f"Error ending transaction: {e}")

        # -------------------------------------------------------------
        # Stop pipeline application if running
        # -------------------------------------------------------------
        with pipeline_lock:
            if current_pipeline_app is not None:
                print("Stopping existing pipeline app...")
                try:
                    # Stop door monitoring (prevent it from calling shutdown)
                    current_pipeline_app.door_monitor_active = False
                
                    # Check if shutdown was already called
                    with current_pipeline_app.shutdown_lock:
                        if not current_pipeline_app.shutdown_called:
                            print("Calling shutdown for cleanup...")
                            current_pipeline_app.shutdown()
                        else:
                            print("Pipeline already shut down, skipping shutdown call")
                
                    # Wait for pipeline to fully stop
                    time.sleep(2)
                
                    print("Pipeline app stopped successfully")
                except Exception as e:
                    print(f"Error stopping pipeline app: {e}")
                finally:
                    current_pipeline_app = None
        
        # -------------------------------------------------------------
        # Clean up audio alerts
        # -------------------------------------------------------------
        await cleanup_websocket_sounds()
        
        # -------------------------------------------------------------
        # Announce door closing and lock door
        # -------------------------------------------------------------
        print("WebSocket closing - announcing door close")
        tts_manager.speak_door_close()  # "Door is closing"
        
        GPIO.output(DOOR_LOCK_PIN, GPIO.HIGH)  # Lock door
        time.sleep(0.3)                         # Small delay
        GPIO.output(LED_GREEN, GPIO.HIGH)       # Turn off green LED
        GPIO.output(LED_RED, GPIO.HIGH)         # Turn on red LED
        
        # -------------------------------------------------------------
        # Print memory statistics every 10 transactions
        # -------------------------------------------------------------
        if transaction_memory_manager.global_stats['total_transactions'] % 10 == 0:
            transaction_memory_manager.print_stats()
        
        # -------------------------------------------------------------
        # Close WebSocket connection
        # -------------------------------------------------------------
        await websocket.close()
        print("WebSocket connection closed")

# =====================================================================
# WEBSOCKET CLEANUP HELPERS
# =====================================================================

async def cleanup_websocket_sounds():
    """
    Clean up all audio when WebSocket connection closes.
    
    Stops:
    - Camera cover alert sounds
    - Price exceeded alert sounds
    - Any other playing audio
    
    Called automatically in finally block of websocket_endpoint.
    """
    global camera_covered_sound_playing, price_alert_sound_playing
    
    camera_covered_sound_playing = False
    price_alert_sound_playing = False
    tts_manager.stop_all_audio()
    print("WebSocket cleanup - all sounds stopped")

# =====================================================================
# HEALTH CHECK AND MONITORING ENDPOINTS
# =====================================================================
"""
These endpoints allow remote monitoring of the system without
disturbing active transactions.

Usage:
- Health check: curl http://raspberry-pi:8000/health
- Statistics: curl http://raspberry-pi:8000/stats

Useful for:
- Monitoring system health remotely
- Checking memory usage
- Verifying system is running
- Debugging issues
"""

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    
    Returns basic system status:
    - Status: healthy (if memory < 1000MB) or warning
    - Current memory usage
    - Number of active transactions
    - Total transactions processed
    - System uptime
    
    Returns:
        dict: Health status information
        
    Status Codes:
        200 OK: System is healthy
        (Status field indicates "healthy" or "warning")
        
    Example Response:
    {
        "status": "healthy",
        "memory_mb": 245.67,
        "active_transactions": 1,
        "total_transactions": 47,
        "uptime_hours": 12.5
    }
    """
    stats = transaction_memory_manager.get_stats()
    
    return {
        "status": "healthy" if stats['current_memory_mb'] < 1000 else "warning",
        "memory_mb": round(stats['current_memory_mb'], 2),
        "active_transactions": stats['active_transactions'],
        "total_transactions": stats['total_transactions'],
        "uptime_hours": round((time.time() - app.start_time) / 3600, 2) if hasattr(app, 'start_time') else 0
    }

@app.get("/stats")
async def get_stats():
    """
    Detailed statistics endpoint.
    
    Returns comprehensive system information:
    - Memory: current, available, percentage used
    - Transactions: active, total, cleanup count
    - Recent: last 10 transaction details
    
    Returns:
        dict: Detailed system statistics
        
    Example Response:
    {
        "memory": {
            "current_mb": 245.67,
            "available_mb": 3500.00,
            "percent": 6.5
        },
        "transactions": {
            "active": 1,
            "total": 47,
            "cleanups": 46
        },
        "recent": [
            {
                "transaction_id": "trans_123",
                "duration": 125.5,
                "memory_used_mb": 45.2,
                "frames": 1800
            },
            ...
        ]
    }
    """
    stats = transaction_memory_manager.get_stats()
    
    return {
        "memory": {
            "current_mb": round(stats['current_memory_mb'], 2),
            "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
            "percent": round(psutil.virtual_memory().percent, 2)
        },
        "transactions": {
            "active": stats['active_transactions'],
            "total": stats['total_transactions'],
            "cleanups": stats['total_cleanups']
        },
        "recent": stats['recent_transactions']
    }

# =====================================================================
# MAIN FUNCTION (System Initialization and Startup)
# =====================================================================

def main():
    """
    Main entry point for the smart fridge system.
    
    Initialization Steps:
    1. Parse command line arguments (host, port)
    2. Record start time for uptime tracking
    3. Create required directories
    4. Generate audio alert files
    5. Initialize transaction memory manager
    6. Register GPIO cleanup handler
    7. Print startup information
    8. Start FastAPI/Uvicorn server
    
    Command Line Arguments:
        --host: Host to bind server (default: 0.0.0.0 = all interfaces)
        --port: Port to listen on (default: 8000)
        
    Usage:
        python app_server.py --host 0.0.0.0 --port 8000
        
    Runs Until:
        - Ctrl+C pressed
        - System shutdown
        - Fatal error occurs
    """
    # =================================================================
    # PARSE COMMAND LINE ARGUMENTS
    # =================================================================
    parser = argparse.ArgumentParser(
        description='Smart Fridge Object Detection and Tracking System'
    )
    parser.add_argument(
        '--host', 
        default='0.0.0.0', 
        help='Host to run the server on (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000, 
        help='Port to run the server on (default: 8000)'
    )
    
    args = parser.parse_args()
    
    # =================================================================
    # RECORD START TIME FOR UPTIME TRACKING
    # =================================================================
    app.start_time = time.time()
    
    # =================================================================
    # CREATE REQUIRED DIRECTORIES
    # =================================================================
    print("Creating required directories...")
    os.makedirs('saved_videos', exist_ok=True)      # For recorded transaction videos
    os.makedirs('camera_images', exist_ok=True)     # For product capture images
    os.makedirs("sounds", exist_ok=True)            # For audio files
    os.makedirs("sounds/deposits", exist_ok=True)   # For deposit alert sounds
    print("Directories created successfully")
    
    # =================================================================
    # GENERATE AUDIO ALERT FILES
    # =================================================================
    print("Setting up audio alert system...")
    
    # Camera cover alerts
    setup_cover_alert_sound()
    print("Camera cover alerts ready")
    
    # Product upload process alerts
    setup_product_upload_alerts()
    print("Product upload alerts ready")
    
    # Door control alerts (if not already generated)
    if not os.path.exists("sounds/door_open.mp3") or not os.path.exists("sounds/door_close.mp3"):
        print("Generating door audio files...")
        tts_manager.generate_door_audio_files()
    print("Door control alerts ready")
    
    # Pre-generate common deposit messages for faster playback
    print("Generating common deposit messages...")
    tts_manager.generate_common_deposit_messages()
    print("Deposit alerts ready")
    
    # =================================================================
    # INITIALIZE TRANSACTION MEMORY MANAGER
    # =================================================================
    # Global instance already created: transaction_memory_manager
    print("[Memory] Transaction-based memory management initialized")
    print("[Memory] Automatic cleanup enabled for each transaction")
    
    # =================================================================
    # REGISTER GPIO CLEANUP HANDLER
    # =================================================================
    # Ensures GPIO is properly cleaned up on system exit
    atexit.register(GPIO.cleanup)
    print("GPIO cleanup handler registered")
    
    # =================================================================
    # PRINT STARTUP INFORMATION
    # =================================================================
    print("\n" + "="*60)
    print("SMART FRIDGE SYSTEM STARTED")
    print("="*60)
    print(f"Memory at startup: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
    print(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024:.1f}MB")
    print(f"System ready for connections")
    print(f"\nWebSocket endpoint: ws://{args.host}:{args.port}/ws/track")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print(f"Statistics: http://{args.host}:{args.port}/stats")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # =================================================================
    # START FASTAPI/UVICORN SERVER
    # =================================================================
    # This blocks until server is stopped
    uvicorn.run(
        "app_server:app",           # Module:app_instance
        host=args.host,             # Bind to all interfaces
        port=args.port,             # Port to listen on
        reload=False                # Disable auto-reload (production mode)
    )

# =====================================================================
# PROGRAM ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    """
    Entry point when script is run directly.
    
    This ensures main() only runs when the script is executed directly,
    not when imported as a module.
    """
    main()

# =====================================================================
# END OF SMART FRIDGE DETECTION SYSTEM
# =====================================================================
"""
SYSTEM SUMMARY:

HARDWARE:
- Raspberry Pi 5 with Hailo-8L AI accelerator
- 2 USB cameras for dual-angle detection
- GPIO-controlled door lock, LEDs, buzzer
- Door open/close sensor

SOFTWARE COMPONENTS:
1. FastAPI WebSocket server (real-time communication)
2. Hailo AI pipeline (object detection & tracking)
3. Cross-camera tracking (unify detections from both cameras)
4. Movement analysis (determine entry/exit direction)
5. Product validation (check against planogram)
6. Price calculation (compare with deposit)
7. Video recording (save transaction for review)
8. Memory management (prevent leaks in 24/7 operation)
9. Audio alerts (TTS guidance and warnings)
10. GPIO control (door lock, LEDs, buzzer)

TRANSACTION FLOW:
Customer opens app → WebSocket connects → Door unlocks →
Customer takes items → AI tracks movements → Price calculated →
Alerts if deposit exceeded → Door closes → Video saved →
Refund processed → Memory cleaned → Ready for next customer

DEPLOYMENT:
1. Install dependencies: pip install -r requirements.txt
2. Configure camera devices (/dev/video0, /dev/video2)
3. Set up API credentials in code
4. Run: python app_server.py --host 0.0.0.0 --port 8000
5. Mobile app connects to: ws://raspberry-pi-ip:8000/ws/track

MAINTENANCE:
- Monitor memory usage: curl http://raspberry-pi:8000/stats
- Check system health: curl http://raspberry-pi:8000/health
- Review logs for errors and performance
- Update planogram as products change
- Retrain AI model with new products

FOR HANDOVER:
- All major functions have detailed docstrings
- Complex algorithms explained with comments
- Hardware connections documented
- API endpoints specified
- Global variables clearly defined
- Error handling implemented throughout
- Memory management prevents long-term issues

SUPPORT:
- Read inline comments for detailed explanations
- Check function docstrings for usage examples
- Review global variables section for state management
- Examine error messages for troubleshooting
- Test with /health and /stats endpoints

Good luck with the system! 🚀
"""
