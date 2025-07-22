import tkinter as tk
from tkinter import font
import time
import socketio
import numpy as np
import math
from PIL import Image, ImageTk
import os
import threading
from datetime import datetime

# --- Socket.IO Client Setup ---
sio = socketio.Client(reconnection_delay_max=5)
FLASK_SERVER_URL = 'http://localhost:5000'

# --- Simulator Constants ---
CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 700
CAR_WIDTH = 60
CAR_HEIGHT = 100
GAME_LOOP_INTERVAL_MS = 50  # ~20 FPS

# --- Car State & Physics ---
car_x, car_y = CANVAS_WIDTH / 2, CANVAS_HEIGHT - 100
car_velocity = 0.0  # pixels per frame
car_angle_deg = 0.0  # 0 = right, 90 = up
steer_angle_deg = 0.0  # steering wheel angle

# --- Physics Parameters ---
MAX_SPEED_KPH = 160  # Maximum speed in km/h
MAX_SPEED = (MAX_SPEED_KPH / 3.6) * (GAME_LOOP_INTERVAL_MS / 1000)  # Convert to pixels/frame
ACCELERATION_RATE = 0.15  # pixels/frame²
BRAKING_RATE = 0.5  # pixels/frame²
FRICTION = 0.97  # velocity multiplier per frame
STEER_RATE = 2.0  # degrees/frame
MAX_STEER_ANGLE = 30.0  # degrees
MIN_SPEED = -10.0  # Reverse speed limit

# --- Road Parameters ---
ROAD_WIDTH = 600
LANE_WIDTH = ROAD_WIDTH / 3
ROAD_CENTER_X = CANVAS_WIDTH / 2
DASH_LENGTH = 20
DASH_GAP = 40
STRIPE_WIDTH = 3

# --- Sensor Parameters (Mobile Phone in Car) ---
PHONE_MOUNT_ANGLE = 15  # degrees tilt from vertical
ACCEL_TO_SENSOR_SCALE = 1.5  # Convert physics to G-forces
GYRO_TO_SENSOR_SCALE = 0.15  # Convert physics to deg/s

# --- Sensor Data & State Variables ---
acc_x, acc_y, acc_z = 0.0, 0.0, 0.0
gyro_x, gyro_y, gyro_z = 0.0, 0.0, 0.0
prev_car_velocity_x, prev_car_velocity_y = 0.0, 0.0
last_update_time = 0
update_interval = 100  # ms

# --- Tkinter Setup ---
root = tk.Tk()
root.title("NeuroDrive Simulator")
canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#4A4A4A")
canvas.pack()

# --- Load Car Image ---
try:
    car_img = Image.new('RGBA', (CAR_WIDTH, CAR_HEIGHT), (0, 0, 255, 255))
    car_img_tk = ImageTk.PhotoImage(car_img)
except Exception as e:
    print(f"Couldn't load car image: {e}")
    car_img_tk = None

# --- UI Elements ---
bold_font = font.Font(family="Arial", size=12, weight="bold")
mono_font = font.Font(family="Consolas", size=10)

# Status indicators
connection_status_text = canvas.create_text(
    10, 10, text="Connecting...", anchor="nw", fill="yellow", font=bold_font
)
behavior_status_text = canvas.create_text(
    CANVAS_WIDTH / 2, 20, text="Behavior: N/A", fill="white", font=bold_font
)
speed_text = canvas.create_text(
    CANVAS_WIDTH - 10, 30, text="Speed: 0 km/h", anchor="ne", fill="white", font=bold_font
)

# Aggressive driving alert
aggressive_alert = canvas.create_text(
    CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2,
    text="!!! AGGRESSIVE DRIVING !!!",
    fill="red", font=("Arial", 24, "bold"), state=tk.HIDDEN
)

# Sensor Data Display
sensor_labels = {}
features_to_display = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
for i, feature in enumerate(features_to_display):
    y_pos = 40 + i * 20
    canvas.create_text(10, y_pos, text=f"{feature}:", anchor="nw", fill="#00FF00", font=mono_font)
    sensor_labels[feature] = canvas.create_text(70, y_pos, text="0.000", anchor="nw", fill="white", font=mono_font)

# --- Drawing Functions ---
def draw_road():
    # Road surface
    canvas.create_rectangle(
        ROAD_CENTER_X - ROAD_WIDTH/2, 0,
        ROAD_CENTER_X + ROAD_WIDTH/2, CANVAS_HEIGHT,
        fill="#333333", outline=""
    )
    
    # Lane markings
    for y in range(0, CANVAS_HEIGHT, DASH_LENGTH + DASH_GAP):
        canvas.create_line(
            ROAD_CENTER_X, y,
            ROAD_CENTER_X, y + DASH_LENGTH,
            fill="yellow", width=STRIPE_WIDTH
        )
        canvas.create_line(
            ROAD_CENTER_X - ROAD_WIDTH/2 + LANE_WIDTH, y,
            ROAD_CENTER_X - ROAD_WIDTH/2 + LANE_WIDTH, y + DASH_LENGTH,
            fill="white", width=STRIPE_WIDTH
        )
        canvas.create_line(
            ROAD_CENTER_X + ROAD_WIDTH/2 - LANE_WIDTH, y,
            ROAD_CENTER_X + ROAD_WIDTH/2 - LANE_WIDTH, y + DASH_LENGTH,
            fill="white", width=STRIPE_WIDTH
        )
    
    # Road edges
    canvas.create_line(
        ROAD_CENTER_X - ROAD_WIDTH/2, 0,
        ROAD_CENTER_X - ROAD_WIDTH/2, CANVAS_HEIGHT,
        fill="white", width=5
    )
    canvas.create_line(
        ROAD_CENTER_X + ROAD_WIDTH/2, 0,
        ROAD_CENTER_X + ROAD_WIDTH/2, CANVAS_HEIGHT,
        fill="white", width=5
    )

car_image_id = None
def draw_car():
    global car_image_id
    if car_img_tk:
        if car_image_id:
            canvas.coords(car_image_id, car_x - CAR_WIDTH/2, car_y - CAR_HEIGHT/2)
        else:
            car_image_id = canvas.create_image(
                car_x - CAR_WIDTH/2, car_y - CAR_HEIGHT/2,
                image=car_img_tk, anchor="nw"
            )
    else:
        points = [
            car_x - CAR_WIDTH/2, car_y - CAR_HEIGHT/4,
            car_x - CAR_WIDTH/4, car_y - CAR_HEIGHT/2,
            car_x + CAR_WIDTH/4, car_y - CAR_HEIGHT/2,
            car_x + CAR_WIDTH/2, car_y - CAR_HEIGHT/4,
            car_x + CAR_WIDTH/2, car_y + CAR_HEIGHT/4,
            car_x + CAR_WIDTH/4, car_y + CAR_HEIGHT/2,
            car_x - CAR_WIDTH/4, car_y + CAR_HEIGHT/2,
            car_x - CAR_WIDTH/2, car_y + CAR_HEIGHT/4,
            car_x - CAR_WIDTH/2, car_y - CAR_HEIGHT/4
        ]
        if car_image_id:
            canvas.coords(car_image_id, points)
        else:
            car_image_id = canvas.create_polygon(points, fill="blue", outline="black")

# --- Keyboard Input ---
keys = {'Up': False, 'Down': False, 'Left': False, 'Right': False}
def on_key_press(event):
    if event.keysym in keys: 
        keys[event.keysym] = True
        # Immediate response to key press
        if event.keysym in ['Left', 'Right']:
            game_loop()
def on_key_release(event):
    if event.keysym in keys: keys[event.keysym] = False

root.bind('<KeyPress>', on_key_press)
root.bind('<KeyRelease>', on_key_release)

# --- Physics Calculations ---
def calculate_sensor_data():
    global acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, prev_car_velocity_x, prev_car_velocity_y
    
    angle_rad = math.radians(car_angle_deg)
    current_velocity_x = car_velocity * math.cos(angle_rad)
    current_velocity_y = car_velocity * math.sin(angle_rad)
    
    delta_vx = current_velocity_x - prev_car_velocity_x
    delta_vy = current_velocity_y - prev_car_velocity_y
    
    cos_rot = math.cos(math.radians(car_angle_deg + PHONE_MOUNT_ANGLE))
    sin_rot = math.sin(math.radians(car_angle_deg + PHONE_MOUNT_ANGLE))
    
    acc_long = (delta_vx * sin_rot - delta_vy * cos_rot) * ACCEL_TO_SENSOR_SCALE
    acc_lat = (delta_vx * cos_rot + delta_vy * sin_rot) * ACCEL_TO_SENSOR_SCALE
    
    acc_x = acc_lat
    acc_y = acc_long
    acc_z = 1.0 + np.random.normal(0, 0.02)
    
    if abs(steer_angle_deg) > 5 and car_velocity > 5:
        acc_x += 0.2 * math.sin(time.time() * 10)
        acc_z += 0.05 * math.sin(time.time() * 15)
    
    gyro_z = -(steer_angle_deg * car_velocity / MAX_SPEED) * GYRO_TO_SENSOR_SCALE
    gyro_y = acc_long * 0.5
    gyro_x = -acc_lat * 0.5
    
    acc_x += np.random.normal(0, 0.02)
    acc_y += np.random.normal(0, 0.02)
    acc_z += np.random.normal(0, 0.01)
    gyro_x += np.random.normal(0, 0.005)
    gyro_y += np.random.normal(0, 0.005)
    gyro_z += np.random.normal(0, 0.01)
    
    prev_car_velocity_x, prev_car_velocity_y = current_velocity_x, current_velocity_y

# --- Main Game Loop ---
def game_loop():
    global car_x, car_y, car_velocity, car_angle_deg, steer_angle_deg, last_update_time
    
    # Physics Update
    if keys['Up']:
        car_velocity = min(car_velocity + ACCELERATION_RATE, MAX_SPEED)
    if keys['Down']:
        car_velocity = max(car_velocity - BRAKING_RATE, MIN_SPEED)
    
    car_velocity *= FRICTION
    if abs(car_velocity) < 0.1: car_velocity = 0

    if keys['Left']:
        steer_angle_deg = min(MAX_STEER_ANGLE, steer_angle_deg + STEER_RATE * (1 - car_velocity/MAX_SPEED*0.7))
    elif keys['Right']:
        steer_angle_deg = max(-MAX_STEER_ANGLE, steer_angle_deg - STEER_RATE * (1 - car_velocity/MAX_SPEED*0.7))
    else:
        steer_angle_deg *= 0.7

    if abs(car_velocity) > 0.5:
        turn_factor = 1.0 - (0.7 * abs(car_velocity) / MAX_SPEED)
        car_angle_deg += steer_angle_deg * car_velocity * 0.03 * turn_factor

    angle_rad = math.radians(car_angle_deg)
    car_x += car_velocity * math.cos(angle_rad)
    car_y += car_velocity * math.sin(angle_rad)

    road_left = ROAD_CENTER_X - ROAD_WIDTH/2 + CAR_WIDTH/2
    road_right = ROAD_CENTER_X + ROAD_WIDTH/2 - CAR_WIDTH/2
    car_x = max(road_left, min(car_x, road_right))
    car_y = max(CAR_HEIGHT/2, min(car_y, CANVAS_HEIGHT - CAR_HEIGHT/2))

    calculate_sensor_data()
    
    speed_kmh = abs(car_velocity) * (1000/GAME_LOOP_INTERVAL_MS) * 3.6
    
    # Update UI
    canvas.itemconfig(speed_text, text=f"Speed: {speed_kmh:.1f} km/h")
    draw_road()
    draw_car()
    
    sensor_values = {
        'AccX': acc_x, 'AccY': acc_y, 'AccZ': acc_z,
        'GyroX': gyro_x, 'GyroY': gyro_y, 'GyroZ': gyro_z
    }
    for f in features_to_display:
        canvas.itemconfig(sensor_labels[f], text=f"{sensor_values[f]:.3f}")

    # Send data to server at controlled intervals
    current_time = int(time.time() * 1000)
    if current_time - last_update_time >= update_interval:
        last_update_time = current_time
        if any(keys.values()) or abs(car_velocity) > 0.1:
            sio.emit('sensor_data', {
                'Timestamp': current_time,
                'AccX': acc_x, 'AccY': acc_y, 'AccZ': acc_z,
                'GyroX': gyro_x, 'GyroY': gyro_y, 'GyroZ': gyro_z,
                'speed': speed_kmh
            })

    root.after(GAME_LOOP_INTERVAL_MS, game_loop)

# --- Socket.IO Event Handlers ---
@sio.event
def connect():
    canvas.after(0, lambda: canvas.itemconfig(connection_status_text, text="Connected", fill="lime"))

@sio.event
def disconnect(reason=None):
    try:
        canvas.itemconfig(connection_status_text, text="Disconnected", fill="red")
        canvas.itemconfig(behavior_status_text, text="Behavior: N/A", fill="white")
    except:
        pass

@sio.event
def connect_error(data):
    canvas.itemconfig(connection_status_text, text="Connection Failed", fill="red")

@sio.on('risk_alert')
def handle_risk_alert(data):
    risk_level = data.get('risk_level', 'N/A')
    colors = {
        'Aggressive': 'red',
        'Normal': 'lime',
        'Slow': 'cyan'
    }
    color = colors.get(risk_level, 'white')
    
    # Update simulator UI
    canvas.itemconfig(behavior_status_text, text=f"Behavior: {risk_level}", fill=color)
    if risk_level == 'Aggressive':
        canvas.itemconfig(aggressive_alert, state=tk.NORMAL)
        root.after(1000, lambda: canvas.itemconfig(aggressive_alert, state=tk.HIDDEN))
    
    # Also send to dashboard
    sio.emit('behavior_update', {
        'risk_level': risk_level,
        'color': color
    })

# --- Main Execution ---
if __name__ == '__main__':
    def connect_thread():
        try:
            sio.connect(FLASK_SERVER_URL, transports=['websocket', 'polling'])
        except Exception as e:
            print(f"Connection error: {e}")
            canvas.after(0, lambda: canvas.itemconfig(connection_status_text, text="Connection Failed", fill="red"))

    threading.Thread(target=connect_thread, daemon=True).start()

    draw_road()
    draw_car()
    game_loop()
    
    try:
        root.mainloop()
    finally:
        if sio.connected:
            sio.disconnect()
        print("Simulator closed.")
        