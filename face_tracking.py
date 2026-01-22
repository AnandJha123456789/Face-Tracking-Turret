import cv2
import serial
import time
import numpy as np
import os
import logging
import sys
import csv  # Changed from scipy.io

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'camera_id': 1,               # 0 for desktop, 1 for laptop/external
    'serial_port': 'COM3',        # CHANGE THIS to match your Arduino port
    'baud_rate': 115200,
    'model_dir': "models/",
    'model_proto': "opencv_face_detector.pbtxt",
    'model_weights': "opencv_face_detector_uint8.pb",
    'conf_threshold': 0.15,       # Face detection confidence
    
    # Motor Settings
    'max_motor_speed': 128,       # Max PWM (0-255)
    'min_motor_speed': 15,        # Min PWM to overcome friction
    'deadzone_threshold': 1.0,    # Degree offset to ignore
    'breaking_exit_val': 256,     # Value to tell Arduino to coast
    
    # Camera Field of View (Half-angles)
    'fov_x_half': 30.0,           # Max Horizontal Offset (Degrees)
    'fov_y_half': 15.0,           # Max Vertical Offset (Degrees)
    
    # PID Control Constants
    'ki': 0.0,                   # Integral gain
    'kd': 0.0,                   # Derivative gain
    
    # UI Colors
    'hud_color': (30, 30, 30),
    'text_color': (200, 200, 200),
    'window_name': "Face Tracking Turret"
}

class MockSerial:
    """Simulates a Serial connection if Arduino is missing."""
    def write(self, data):
        pass
    def readline(self):
        return b''
    def close(self):
        pass

class FaceTrackingTurret:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Calculate Kp dynamically
        self.kp = CONFIG['max_motor_speed'] / CONFIG['fov_x_half']
        self.ki = CONFIG['ki']
        self.kd = CONFIG['kd']
        
        logger.info(f"PID Initialized: Kp={self.kp:.2f}, Ki={self.ki}, Kd={self.kd}")

        self.init_model()
        self.init_serial()
        self.cap = cv2.VideoCapture(CONFIG['camera_id'])
        
        if not self.cap.isOpened():
            logger.error(f"Could not open camera {CONFIG['camera_id']}. Please use camera ID 0 for built-in webcam, camera ID 1 for external camera.")
            sys.exit(1)

        # PID State Variables
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.last_pid_time = time.time()
        
        # Toggle state for showing all faces
        self.show_all_faces = False
        
        # Data logging
        self.history = {
            'time': [],
            'motor_speed': [],
            'error_angle': [],
            'p_term': [],
            'i_term': [],
            'd_term': []
        }
        self.start_time = time.time()

    def init_model(self):
        """Load DNN Face Detection Model."""
        model_path = os.path.join(self.script_dir, f"{CONFIG['model_dir']}{CONFIG['model_weights']}")
        config_path = os.path.join(self.script_dir, f"{CONFIG['model_dir']}{CONFIG['model_proto']}")
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            logger.error(f"Model files not found in {self.script_dir}")
            sys.exit(1)
            
        self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        logger.info("DNN Model Loaded.")

    def init_serial(self):
        """Initialize Serial or fallback to Mock mode."""
        try:
            self.ser = serial.Serial(CONFIG['serial_port'], CONFIG['baud_rate'], timeout=0.01)
            time.sleep(2) # Allow Arduino to reset
            logger.info(f"Connected to Arduino on {CONFIG['serial_port']}")
            self.arduino_connected = True
        except (serial.SerialException, FileNotFoundError):
            logger.warning("Arduino not found. Running in SIMULATION MODE.")
            self.ser = MockSerial()
            self.arduino_connected = False

    def pixel_to_degrees(self, x, y, frame_w, frame_h):
        center_x = frame_w / 2.0
        center_y = frame_h / 2.0
        
        norm_x = (x - center_x) / center_x
        norm_y = (y - center_y) / center_y
        
        deg_x = norm_x * CONFIG['fov_x_half']
        deg_y = -1 * (norm_y * CONFIG['fov_y_half']) 
        
        return deg_x, deg_y

    def calculate_pid_output(self, error_angle):
        current_time = time.time()
        dt = current_time - self.last_pid_time
        if dt <= 0: dt = 0.001
        
        if abs(error_angle) <= CONFIG['deadzone_threshold']:
            self.last_pid_time = current_time
            self.integral_error = 0 
            return 0, 0, 0, 0

        P = self.kp * error_angle
        
        self.integral_error += error_angle * dt
        i_limit = CONFIG['max_motor_speed'] / 2
        self.integral_error = max(min(self.integral_error, i_limit), -i_limit)
        I = self.ki * self.integral_error
        
        delta_error = error_angle - self.prev_error
        D = self.kd * (delta_error / dt)
        
        output = P + I + D
        self.prev_error = error_angle
        self.last_pid_time = current_time
        
        if output > 0:
            speed = max(CONFIG['min_motor_speed'], min(CONFIG['max_motor_speed'], output))
        elif output < 0:
            speed = min(-CONFIG['min_motor_speed'], max(-CONFIG['max_motor_speed'], output))
        else:
            speed = 0
            
        return int(speed), P, I, D

    def draw_hud(self, frame, faces_data, error_angle, motor_speed, pid_terms):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 1. Draw Faces
        for face in faces_data:
            (x, y, bw, bh) = face['box']
            deg_x, deg_y = face['deg']
            is_target = face['is_target']
            
            # --- COLOR SELECTION ---
            if is_target:
                # Target Logic: Red if locked, Orange if tracking
                if abs(deg_x) <= CONFIG['deadzone_threshold']:
                    color = (0, 0, 255)     # Red
                    thickness = 3
                else:
                    color = (0, 100, 200)   # Orange
                    thickness = 2
                
                # Text style for Target
                text_bg_color = color
                text_font_color = (0, 0, 0) # Black text
                
            else:
                # Non-Target Logic: Dark Grey
                color = (60, 60, 60)        # Dark Grey
                thickness = 1
                
                # Text style for Non-Target
                text_bg_color = (200, 200, 200) # Light grey bg to make dark text visible
                text_font_color = (60, 60, 60)  # Dark grey text
            
            # Draw Box
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, thickness)
            
            # --- COORDINATES TEXT ---
            coord_text = f"X:{deg_x:.1f} Y:{deg_y:.1f}"
            text_size = cv2.getTextSize(coord_text, font, 0.5, 1)[0]
            
            # Draw background for text
            cv2.rectangle(frame, (x, y - 20), (x + text_size[0], y), text_bg_color, -1)
            # Draw text
            cv2.putText(frame, coord_text, (x, y - 5), font, 0.5, text_font_color, 1)

            # Note: Removed "TARGET" word as requested

        # 2. Create Sidebar
        sidebar_w = 280
        canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[:, w:] = CONFIG['hud_color']
        
        # 3. Add Text Stats
        y_start = 40
        lh = 30
        
        header = "ARDUINO: ON" if self.arduino_connected else "ARDUINO: OFF"
        header_col = (0, 255, 0) if self.arduino_connected else (0, 0, 255)
        cv2.putText(canvas, header, (w + 10, y_start), font, 0.7, header_col, 2)
        
        cv2.putText(canvas, f"Error (deg): {error_angle:.2f}", (w + 10, y_start + lh), font, 0.6, CONFIG['text_color'], 1)
        cv2.putText(canvas, f"Motor PWM: {motor_speed}", (w + 10, y_start + lh*2), font, 0.6, CONFIG['text_color'], 1)
        
        p_val, i_val, d_val = pid_terms
        cv2.putText(canvas, "PID Terms:", (w + 10, int(y_start + lh*3.5)), font, 0.6, (255, 255, 0), 1)
        cv2.putText(canvas, f"P: {p_val:.2f}", (w + 20, int(y_start + lh*4.5)), font, 0.5, CONFIG['text_color'], 1)
        cv2.putText(canvas, f"I: {i_val:.2f}", (w + 20, int(y_start + lh*5.5)), font, 0.5, CONFIG['text_color'], 1)
        cv2.putText(canvas, f"D: {d_val:.2f}", (w + 20, int(y_start + lh*6.5)), font, 0.5, CONFIG['text_color'], 1)

        # 4. Visual Speed Bar
        bar_center_x = w + sidebar_w // 2
        bar_y_start = int(y_start + lh * 8)
        bar_height = 150
        bar_width = 40
        
        cv2.rectangle(canvas, (bar_center_x - bar_width//2, bar_y_start), 
                      (bar_center_x + bar_width//2, bar_y_start + bar_height), (50,50,50), -1)
        mid_y = bar_y_start + bar_height // 2
        cv2.line(canvas, (bar_center_x - bar_width, mid_y), (bar_center_x + bar_width, mid_y), (255,255,255), 1)

        if motor_speed != 0:
            fill_h = int((abs(motor_speed) / CONFIG['max_motor_speed']) * (bar_height / 2))
            if motor_speed > 0:
                cv2.rectangle(canvas, (bar_center_x - bar_width//2, mid_y - fill_h), 
                              (bar_center_x + bar_width//2, mid_y), (0, 255, 0), -1)
            else:
                cv2.rectangle(canvas, (bar_center_x - bar_width//2, mid_y), 
                              (bar_center_x + bar_width//2, mid_y + fill_h), (0, 0, 255), -1)

        return canvas

    def run(self):
        logger.info("Starting Face Tracking...")
        logger.info("Controls: 'q' to Quit, 't' to Toggle All/Target faces.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame.")
                break

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
            self.net.setInput(blob)
            detections = self.net.forward()

            max_face_size = 0
            max_face_index = -1
            
            # Pass 1: Find largest face (Target)
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIG['conf_threshold']:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face_size = (endX - startX) * (endY - startY)
                    
                    if face_size > max_face_size:
                        max_face_size = face_size
                        max_face_index = i

            # Pass 2: Process Data
            faces_to_draw = []
            target_error_x = 0
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIG['conf_threshold']:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    bw = endX - startX
                    bh = endY - startY
                    
                    center_x = startX + bw / 2.0
                    center_y = startY + bh / 2.0
                    
                    deg_x, deg_y = self.pixel_to_degrees(center_x, center_y, w, h)
                    
                    is_target = (i == max_face_index)
                    
                    if is_target:
                        target_error_x = deg_x
                    
                    # Add to drawing list if it's the target OR if we are showing all faces
                    if is_target or self.show_all_faces:
                        faces_to_draw.append({
                            'box': (startX, startY, bw, bh),
                            'deg': (deg_x, deg_y),
                            'is_target': is_target
                        })

            if max_face_index != -1:
                motor_speed, P, I, D = self.calculate_pid_output(target_error_x)
            else:
                motor_speed = 0
                P, I, D = 0, 0, 0

            if self.arduino_connected:
                try:
                    self.ser.write(f"{motor_speed}\n".encode())
                except Exception as e:
                    logger.error(f"Serial Error: {e}")

            # Logging Data
            curr_time = time.time() - self.start_time
            self.history['time'].append(curr_time)
            self.history['motor_speed'].append(motor_speed)
            self.history['error_angle'].append(target_error_x)
            self.history['p_term'].append(P)
            self.history['i_term'].append(I)
            self.history['d_term'].append(D)

            # Display
            hud = self.draw_hud(frame, faces_to_draw, target_error_x, motor_speed, (P, I, D))
            cv2.imshow(CONFIG['window_name'], hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                self.show_all_faces = not self.show_all_faces
                logger.info(f"Show All Faces: {self.show_all_faces}")

        self.cleanup()

    def cleanup(self):
        logger.info("Shutting down...")
        if self.arduino_connected:
            try:
                self.ser.write(f"{CONFIG['breaking_exit_val']}\n".encode())
                self.ser.close()
            except Exception as e:
                logger.error(f"Error closing serial: {e}")
            
        self.cap.release()
        cv2.destroyAllWindows()
        
        # CSV Saving Logic
        try:
            file_name = os.path.join(self.script_dir, 'tracking_data.csv')
            with open(file_name, mode='w', newline='') as f:
                writer = csv.writer(f)
                
                # Write Header
                keys = list(self.history.keys())
                writer.writerow(keys)
                
                # Write Data (Zip columns into rows)
                rows = zip(*[self.history[k] for k in keys])
                writer.writerows(rows)
                
            logger.info(f"Data saved to {file_name}")
        except Exception as e:
            logger.error(f"Failed to save CSV data: {e}")

if __name__ == "__main__":
    app = FaceTrackingTurret()
    app.run()