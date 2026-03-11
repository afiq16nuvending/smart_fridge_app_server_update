# =====================================================================
# HOW TO CONVERT YOUR SMART FRIDGE TO POST-PROCESSING
# =====================================================================
#
# Your original code is ~5500 lines. Only ~300 lines need to change.
# Everything else (detection_callback, cross-camera tracking, movement
# analysis, planogram, TTS, memory management) stays EXACTLY the same.
#
# WHAT STAYS THE SAME (don't touch these):
#   - All imports
#   - All GPIO setup
#   - All global tracking structures  
#   - compute_color_for_labels()
#   - draw_trail()
#   - draw_counts()
#   - draw_zone()
#   - handle_alert_state()
#   - calculate_total_price_and_control_buzzer()
#   - is_frame_dark() + setup_cover_alert_sound() + handle_cover_alert()
#   - check_door_status()
#   - stream_video_to_api()
#   - monitor_and_send_videos() + is_file_complete_enhanced()
#   - WebSocketDataManager class
#   - TrackingData class
#   - HailoDetectionCallback class (except get_fallback_pipeline_string)
#   - get_global_track_id()
#   - cleanup_inactive_tracks()
#   - analyze_movement_direction()
#   - detection_callback() <-- THE HEART OF YOUR SYSTEM, UNCHANGED
#   - Product capture system (capture_images, upload_images_to_api, etc.)
#   - TransactionMemoryManager class
#   - TTSManager class
#   - Health check endpoints
#   - main() function
#
# =====================================================================
# CHANGES NEEDED (3 things):
# =====================================================================
#
# 1. ADD: DualCameraRecorder class (new, ~120 lines)
# 2. MODIFY: get_fallback_pipeline_string() in HailoDetectionCallback
#    - Change v4l2src to filesrc for both cameras
# 3. MODIFY: HailoDetectionApp class
#    - Remove door monitoring (door already closed)
#    - Add EOS handling (video file ends = pipeline done)
# 4. REPLACE: run_tracking() function
#    - Old: start pipeline with live cameras
#    - New: record → stop → run pipeline on files → get results
# 5. ADD: reset_tracking_state() helper function
#
# =====================================================================


# =====================================================================
# CHANGE 1: ADD THIS CLASS (put it before HailoDetectionApp)
# =====================================================================

class DualCameraRecorder:
    """
    Records video from two USB cameras simultaneously.
    Used DURING the transaction (no AI, just recording).
    
    Output:
        saved_videos/cam0_YYYYMMDD_HHMMSS_[transaction_id].avi
        saved_videos/cam1_YYYYMMDD_HHMMSS_[transaction_id].avi
    """

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
        self.cam0_video_path = os.path.join(
            self.video_dir, f"cam0_{timestamp}_{transaction_id}.avi")
        self.cam1_video_path = os.path.join(
            self.video_dir, f"cam1_{timestamp}_{transaction_id}.avi")

    def _record_camera(self, device_id, output_path, camera_name):
        """Record from one camera in a thread."""
        global camera_covered, cover_alert_thread
        cap = None
        writer = None
        frame_count = 0
        fps_start_time = None
        fps_calculated = False
        actual_fps = 15.0

        try:
            cap = cv2.VideoCapture(device_id)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
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

                # Camera cover detection still active
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
            if writer: writer.release()
            if cap: cap.release()
            print(f"[Recording] {camera_name}: {frame_count} frames saved")

    def start_recording(self):
        """Start both cameras recording."""
        self.is_recording = True
        self.shutdown_event.clear()
        self.cam0_thread = threading.Thread(
            target=self._record_camera,
            args=(self.cam0_device, self.cam0_video_path, "Camera 0"), daemon=True)
        self.cam1_thread = threading.Thread(
            target=self._record_camera,
            args=(self.cam1_device, self.cam1_video_path, "Camera 1"), daemon=True)
        self.cam0_thread.start()
        self.cam1_thread.start()
        print(f"[Recording] Dual camera recording started")

    def stop_recording(self):
        """Stop recording, return file paths."""
        self.shutdown_event.set()
        self.is_recording = False
        if self.cam0_thread and self.cam0_thread.is_alive():
            self.cam0_thread.join(timeout=5)
        if self.cam1_thread and self.cam1_thread.is_alive():
            self.cam1_thread.join(timeout=5)
        print(f"[Recording] Stopped. Cam0: {self.cam0_frame_count}f, Cam1: {self.cam1_frame_count}f")
        return self.cam0_video_path, self.cam1_video_path


# =====================================================================
# CHANGE 2: MODIFY get_fallback_pipeline_string() in HailoDetectionCallback
# 
# Replace the v4l2src lines with filesrc lines.
# Everything else in the pipeline string stays IDENTICAL.
# =====================================================================
#
# In your HailoDetectionCallback class, replace get_fallback_pipeline_string():

    def get_fallback_pipeline_string(self):
        """
        MODIFIED: Read from video files instead of live cameras.
        
        ONLY CHANGE: 
          v4l2src device=/dev/video0 ! image/jpeg ... ! jpegdec
            becomes
          filesrc location=<path> ! decodebin
          
        EVERYTHING ELSE IS IDENTICAL:
          Same hailonet, same .hef, same .so, same hailotracker,
          same identity_callback_0/1, same compositor
        """
        cam0_video = os.environ.get('POSTPROCESS_CAM0_VIDEO', '')
        cam1_video = os.environ.get('POSTPROCESS_CAM1_VIDEO', '')
        
        return (
            # ===== CORE INFERENCE (IDENTICAL TO ORIGINAL) =====
            "hailoroundrobin mode=0 name=fun ! "
            "queue name=hailo_pre_infer_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "hailonet hef-path=resources/ai_model.hef batch-size=2 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! "
            "queue name=hailo_postprocess0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "hailofilter function-name=filter_letterbox so-path=/home/afiq/hailo-rpi5-examples/basic_pipelines/../resources/libyolo_hailortpp_postprocess.so config-path=resources/labels.json qos=false ! "
            "queue name=hailo_track0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "hailotracker name=hailo_tracker class-id=-1 kalman-dist-thr=0.8 iou-thr=0.9 init-iou-thr=0.7 keep-new-frames=1 keep-tracked-frames=1 keep-lost-frames=1 keep-past-metadata=true ! "
            "hailostreamrouter name=sid src_0::input-streams=\"<sink_0>\" src_1::input-streams=\"<sink_1>\" "
            
            # ===== DISPLAY (IDENTICAL TO ORIGINAL) =====
            "compositor name=comp start-time-selection=0 sink_0::xpos=0 sink_0::ypos=0 sink_1::xpos=350 sink_1::ypos=0 ! "
            "queue name=hailo_video_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "videoconvert ! "
            "queue name=hailo_display_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "fpsdisplaysink video-sink=ximagesink name=hailo_display sync=false text-overlay=true "
            
            # ===== CAMERA 0: filesrc INSTEAD OF v4l2src =====
            # OLD: "v4l2src device=/dev/video0 name=source_0 ! "
            #      "image/jpeg, width=640, height=360, framerate=25/1 ! "
            #      "jpegdec ! "
            # NEW:
            f"filesrc location={cam0_video} ! "
            "decodebin ! "
            # REST IS IDENTICAL:
            "queue name=source_scale_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "videoscale name=source_videoscale_0 n-threads=2 ! "
            "queue name=source_convert_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "videoconvert n-threads=3 name=source_convert_0 qos=false ! "
            "video/x-raw, format=RGB, pixel-aspect-ratio=1/1 ! "
            "queue name=inference_wrapper_input_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "fun.sink_0 "
            
            # ===== CAMERA 0 OUTPUT (IDENTICAL TO ORIGINAL) =====
            "sid.src_0 ! "
            "queue name=identity_callback_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "identity name=identity_callback_0 ! "
            "queue name=hailo_draw_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "hailooverlay ! "
            "videoscale n-threads=8 ! "
            "video/x-raw,width=640,height=360 ! "
            "queue name=comp_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "comp.sink_0 "
            
            # ===== CAMERA 1: filesrc INSTEAD OF v4l2src =====
            # OLD: "v4l2src device=/dev/video2 name=source_2 ! "
            #      "image/jpeg, width=640, height=360, framerate=25/1 ! "
            #      "jpegdec ! "
            # NEW:
            f"filesrc location={cam1_video} ! "
            "decodebin ! "
            # REST IS IDENTICAL:
            "queue name=source_scale_q_2 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "videoscale name=source_videoscale_2 n-threads=2 ! "
            "queue name=source_convert_q_2 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "videoconvert n-threads=3 name=source_convert_2 qos=false ! "
            "video/x-raw, format=RGB, pixel-aspect-ratio=1/1 ! "
            "queue name=inference_wrapper_input_q_2 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
            "fun.sink_1 "
            
            # ===== CAMERA 1 OUTPUT (IDENTICAL TO ORIGINAL) =====
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


# =====================================================================
# CHANGE 3: MODIFY HailoDetectionApp class
# 
# Two changes:
# a) Remove door monitoring (door is already closed during post-processing)
# b) Handle EOS (End of Stream) = video file finished = shutdown
# =====================================================================
#
# In HailoDetectionApp.__init__, REMOVE these lines:
#     self.door_monitor_active = True
#     self.door_monitor_thread = threading.Thread(target=self.monitor_door)
#     self.door_monitor_thread.daemon = True
#     ...
#     self.door_monitor_thread.start()
#
# And REMOVE the monitor_door() method entirely.
#
# In bus_call(), CHANGE the EOS handler to call shutdown:

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            # VIDEO FILES FINISHED - trigger shutdown
            print("[PostProcess] End-of-stream - videos fully processed")
            self.shutdown()  # <-- THIS IS THE KEY CHANGE
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
        return True

# In run(), you can optionally remove the display_process for headless mode:
#     # if self.use_frame:
#     #     display_process = multiprocessing.Process(...)
#     #     display_process.start()


# =====================================================================
# CHANGE 4: ADD reset_tracking_state() helper
# Put this anywhere before run_tracking()
# =====================================================================

def reset_tracking_state():
    """Reset all global tracking data between transactions."""
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
    camera_movement_history = {
        0: defaultdict(lambda: deque(maxlen=5)),
        1: defaultdict(lambda: deque(maxlen=5))
    }
    camera_bbox_area_history = {
        0: defaultdict(lambda: deque(maxlen=5)),
        1: defaultdict(lambda: deque(maxlen=5))
    }
    print("[PostProcess] Tracking state reset for new transaction")


# =====================================================================
# CHANGE 5: REPLACE run_tracking() function
#
# This is the main change. The new flow:
#   1. Same: wait for start_preview message
#   2. Same: extract deposit, machine_id, etc.
#   3. NEW:  start DualCameraRecorder (record only, no AI)
#   4. NEW:  wait for door to close
#   5. NEW:  stop recording
#   6. NEW:  set video paths in environment
#   7. NEW:  reset tracking state
#   8. SAME: create HailoDetectionCallback (loads planogram)
#   9. SAME: create HailoDetectionApp (but reads from files now)
#  10. SAME: app.run() - blocks until EOS (same detection_callback runs!)
#  11. NEW:  get results, calculate price, send to mobile app
# =====================================================================

async def run_tracking(websocket: WebSocket):
    """POST-PROCESSING VERSION of run_tracking."""
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

    # =================================================================
    # PHASE 1: WAIT FOR START MESSAGE (SAME AS ORIGINAL)
    # =================================================================
    while True:
        try:
            message_text = await websocket.receive_text()
            print(f"Received message: {message_text}")
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

    # =================================================================
    # PHASE 2: DOOR CONTROL (SAME AS ORIGINAL)
    # =================================================================
    if unlock_data == 1:
        readyToProcess = True
        unlock_data = 0

    try:
        # =============================================================
        # MODE 1: PRODUCT UPLOAD (SAME AS ORIGINAL, NO CHANGES)
        # =============================================================
        if isinstance(message, dict) and message.get('action') == 'product_upload':
            # ... keep your entire product_upload code unchanged ...
            done = True
            # (same capture_images, upload, delete logic)
            pass

        # =============================================================
        # MODE 2: DETECTION (POST-PROCESSING - THIS IS THE NEW PART)
        # =============================================================
        else:
            if transaction_id:
                transaction_memory_manager.start_transaction(transaction_id)

            # ---------------------------------------------------------
            # STEP 1: START RECORDING (cameras only, NO AI)
            # ---------------------------------------------------------
            recorder = DualCameraRecorder(
                transaction_id=transaction_id,
                cam0_device=0,
                cam1_device=2
            )
            recorder.start_recording()

            await websocket.send_json({
                "status": "recording",
                "message": "Cameras recording. Take your items."
            })

            # ---------------------------------------------------------
            # STEP 2: WAIT FOR DOOR TO CLOSE
            # ---------------------------------------------------------
            door_monitor_active = True
            done = True
            print("[Transaction] Waiting for door to close...")

            while door_monitor_active:
                door_sw = GPIO.input(DOOR_SWITCH_PIN)
                if door_sw == 0:
                    print("[Transaction] Door closed!")
                    door_monitor_active = False
                    break
                await asyncio.sleep(0.1)

            # ---------------------------------------------------------
            # STEP 3: STOP RECORDING
            # ---------------------------------------------------------
            cam0_path, cam1_path = recorder.stop_recording()

            await websocket.send_json({
                "status": "processing",
                "message": "Door closed. Analyzing your transaction..."
            })

            # ---------------------------------------------------------
            # STEP 4: RESET TRACKING STATE
            # ---------------------------------------------------------
            reset_tracking_state()

            # ---------------------------------------------------------
            # STEP 5: SET VIDEO PATHS FOR PIPELINE
            # ---------------------------------------------------------
            os.environ['POSTPROCESS_CAM0_VIDEO'] = cam0_path
            os.environ['POSTPROCESS_CAM1_VIDEO'] = cam1_path

            # ---------------------------------------------------------
            # STEP 6: CREATE CALLBACK (SAME AS ORIGINAL)
            # This loads the planogram, sets up validation, etc.
            # ---------------------------------------------------------
            callback = HailoDetectionCallback(
                websocket, deposit, machine_id,
                machine_identifier, user_id, transaction_id
            )

            # ---------------------------------------------------------
            # STEP 7: RUN HAILO PIPELINE ON SAVED VIDEOS
            # Same HailoDetectionApp, same detection_callback!
            # Only difference: source is filesrc instead of v4l2src
            # Pipeline will process all frames then hit EOS and stop.
            # ---------------------------------------------------------
            print(f"\n[PostProcess] Running pipeline on saved videos...")
            print(f"  Cam0: {cam0_path}")
            print(f"  Cam1: {cam1_path}")

            pipeline_app = HailoDetectionApp(detection_callback, callback)

            with pipeline_lock:
                if current_pipeline_app is not None:
                    current_pipeline_app.shutdown()
                    time.sleep(2)
                current_pipeline_app = pipeline_app

            pipeline_app.run()  # BLOCKS until both videos processed (EOS)

            # ---------------------------------------------------------
            # STEP 8: GET RESULTS (from detection_callback data)
            # detection_callback already populated all the counters,
            # validated_products, etc. via the WebSocketDataManager.
            # ---------------------------------------------------------
            final_data = callback.tracking_data.websocket_data_manager.get_current_data()
            total_price = calculate_total_price_and_control_buzzer(
                final_data, deposit, None
            )
            refund = max(0, deposit - total_price)

            # ---------------------------------------------------------
            # STEP 9: SEND RESULTS TO MOBILE APP
            # ---------------------------------------------------------
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

            print(f"\n{'='*60}")
            print(f"RESULTS: Price=${total_price:.2f}, Refund=${refund:.2f}")
            print(f"{'='*60}\n")

            # ---------------------------------------------------------
            # STEP 10: UPLOAD VIDEOS (background)
            # ---------------------------------------------------------
            for vpath in [cam0_path, cam1_path]:
                if vpath and os.path.exists(vpath):
                    dname = os.path.splitext(os.path.basename(vpath))[0]
                    threading.Thread(
                        target=stream_video_to_api,
                        args=(vpath, dname, transaction_id,
                              machine_id, user_id, machine_identifier),
                        daemon=True
                    ).start()

            if transaction_id:
                transaction_memory_manager.end_transaction(transaction_id)

    except Exception as e:
        print(f"Error: {e}")
        if transaction_id:
            try: transaction_memory_manager.end_transaction(transaction_id)
            except: pass

    finally:
        try:
            await websocket.send_json({"status": "stopped", "message": "Complete"})
        except: pass
        if recorder and recorder.is_recording:
            recorder.stop_recording()
        # ... rest of cleanup same as original ...


# =====================================================================
# CHANGE 6: In detection_callback(), REMOVE the buzzer/LED per-frame
#
# In STEP 13, replace calculate_total_price_and_control_buzzer call
# with nothing (or just skip it). Price is calculated ONCE after
# the pipeline finishes, not per-frame.
# =====================================================================
#
# In detection_callback(), change STEP 13 from:
#
#     # STEP 13: CALCULATE PRICE AND TRIGGER ALERTS
#     current_data = user_data.tracking_data.websocket_data_manager.get_current_data()
#     deposit = user_data.deposit
#     total_price = calculate_total_price_and_control_buzzer(
#         current_data, deposit, label
#     )
#
# To:
#
#     # STEP 13: SKIP PER-FRAME PRICE (calculated after pipeline finishes)
#     # Price and alerts are handled in run_tracking() after pipeline.run()
#     pass
#
# This is optional - leaving it won't break anything, it just does
# unnecessary work per frame. The buzzer won't trigger since the door
# is closed anyway.


# =====================================================================
# SUMMARY OF ALL CHANGES
# =====================================================================
#
# 1. ADD DualCameraRecorder class (~120 lines)
#    - Records both cameras to .avi files
#    
# 2. MODIFY get_fallback_pipeline_string() 
#    - Replace 3 lines per camera (v4l2src + jpeg + jpegdec) 
#    - With 2 lines per camera (filesrc + decodebin)
#    - Read paths from environment variables
#    
# 3. MODIFY HailoDetectionApp
#    - Remove door monitoring
#    - EOS in bus_call triggers shutdown
#    
# 4. ADD reset_tracking_state() (~20 lines)
#    
# 5. REPLACE run_tracking() body
#    - Old: start pipeline → monitor door → shutdown on close
#    - New: start recorder → monitor door → stop recorder → 
#           set file paths → start pipeline → wait for EOS →
#           get results → send to app
#
# 6. OPTIONAL: Remove per-frame price calculation in detection_callback
#
# TOTAL: ~200 lines of new code, ~20 lines modified, ~10 lines removed
# The other ~5300 lines stay EXACTLY the same.
# =====================================================================
