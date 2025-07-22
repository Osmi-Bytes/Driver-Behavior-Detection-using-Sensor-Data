import os
import threading
import sqlite3
import numpy as np
import subprocess
import sys
import pandas as pd
from datetime import datetime

# Add these global variables
session_start_time = None
session_active = False

# CRITICAL: Apply monkey patching for eventlet FIRST.
import eventlet
eventlet.monkey_patch()

# IMPORTANT: These environment variables must be set BEFORE importing tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model

# --- Flask and SocketIO Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_and_unique_key_here_for_security'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Global variables ---
simulator_process = None
active_session = False
session_start_time = None
session_start_time = None
session_active = False


# --- Database Configuration ---
DATABASE = 'instance/driving_behavior.db'

def init_db():
    try:
        os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS driving_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                acc_x REAL,
                acc_y REAL,
                acc_z REAL,
                gyro_x REAL,
                gyro_y REAL,
                gyro_z REAL,
                speed REAL,
                predicted_class INTEGER,
                risk_level TEXT
            )
        ''')
            
        conn.commit()
        conn.close()
        print(f"INFO: Database '{DATABASE}' initialized or already exists.")
    except Exception as e:
        print(f"ERROR: Failed to initialize database: {e}")

def clear_driving_logs():
    """Clears all records from the driving_log table for a new session."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM driving_log')
        conn.commit()
        conn.close()
        print("INFO: Previous driving logs cleared for new session.")
    except Exception as e:
        print(f"ERROR: Failed to clear database: {e}")

# --- TensorFlow Model Loading ---
lstm_model = None
try:
    if os.path.exists("models/lstm_model.h5"):
        lstm_model = load_model("models/lstm_model.h5")
        print("INFO: LSTM model loaded successfully.")
    else:
        print("INFO: 'models/lstm_model.h5' not found. Running without predictions.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load LSTM model: {e}")
    print("The application will run, but predictions will NOT be available.")

# --- In-memory buffer for Time Series (LSTM input) ---
sequence_buffer = []
SEQUENCE_LENGTH = 5
NUM_FEATURES = 6

# --- Flask Routes ---
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/run_simulator')
def run_simulator():
    global simulator_process, active_session, session_start_time
    try:
        if simulator_process is None or simulator_process.poll() is not None:
            python_path = sys.executable
            simulator_process = subprocess.Popen([python_path, "simulator.py"])
            active_session = True
            session_start_time = datetime.now()
            return jsonify({
                "status": "success", 
                "message": "Simulator started successfully",
                "start_time": session_start_time.strftime('%Y-%m-%d %H:%M:%S')
            })
        return jsonify({
            "status": "error", 
            "message": "Simulator is already running"
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Failed to start simulator: {str(e)}"
        })

@app.route('/reset_session')
def reset_session():
    global active_session, session_start_time
    clear_driving_logs()
    active_session = False
    session_start_time = None
    return jsonify({
        "status": "success", 
        "message": "Session reset successfully"
    })

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    print(f"INFO: Client connected. SID: {request.sid}")
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"INFO: Client disconnected. SID: {request.sid}")

@socketio.on('reset_session')
def handle_reset_session():
    global session_start_time, session_active
    clear_driving_logs()
    session_start_time = datetime.now()
    session_active = True
    emit('session_reset', {'status': 'success', 'start_time': session_start_time.isoformat()})
    
@socketio.on('sensor_data')
#def handle_sensor_data(data_point):
#    global sequence_buffer
#
#    timestamp = data_point.get('Timestamp')
#    features = [
#        data_point.get('AccX'), data_point.get('AccY'), data_point.get('AccZ'),
#        data_point.get('GyroX'), data_point.get('GyroY'), data_point.get('GyroZ')
#    ]
#    speed = data_point.get('speed', 0)
def handle_sensor_data(data_point):
    global sequence_buffer, session_start_time, session_active
    
    if not session_active:
        session_start_time = datetime.now()
        session_active = True

    # Ensure all required fields are present
    required_fields = ['Timestamp', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'speed']
    if not all(field in data_point for field in required_fields):
        print("ERROR: Missing fields in sensor data")
        return

    timestamp = data_point['Timestamp']
    features = [
        data_point['AccX'], data_point['AccY'], data_point['AccZ'],
        data_point['GyroX'], data_point['GyroY'], data_point['GyroZ']
    ]
    speed = data_point['speed']

    if any(f is None for f in features) or timestamp is None:
        return

    current_features = np.array(features, dtype=np.float32)
    sequence_buffer.append(current_features)

    if len(sequence_buffer) > SEQUENCE_LENGTH:
        sequence_buffer.pop(0)

    predicted_class = 0
    risk_level = "Collecting Data"

    if len(sequence_buffer) == SEQUENCE_LENGTH and lstm_model:
        try:
            input_sequence = np.array([sequence_buffer])
            prediction = lstm_model.predict(input_sequence, verbose=0)
            predicted_class = int(np.argmax(prediction) + 1)

            risk_level = {
                1: "Aggressive",
                2: "Normal",
                3: "Slow"
            }.get(predicted_class, "Invalid Class")
        except Exception as e:
            print(f"ERROR: Error during LSTM prediction: {e}")
            risk_level = "Error"

    try:
        conn = sqlite3.connect(DATABASE)
        cur = conn.cursor()
        cur.execute('''INSERT INTO driving_log
                      (timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed, predicted_class, risk_level)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (timestamp, *features, speed, predicted_class, risk_level))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"ERROR: Error storing data to database: {e}")

    emit('update', {
        'timestamp': timestamp,
        'speed': speed,
        'class': predicted_class,
        'risk_level': risk_level,
        'sensor_data': {
            'AccX': features[0],
            'AccY': features[1],
            'AccZ': features[2],
            'GyroX': features[3],
            'GyroY': features[4],
            'GyroZ': features[5]
        }
    })
    emit('risk_alert', {'risk_level': risk_level})

@socketio.on('generate_driving_report')
def handle_generate_driving_report(data=None):
    driver_name = "Osmi" if data is None else data.get('driver_name', 'Osmi')
    print(f"INFO: Generating driving report for {driver_name}...")
    
    try:
        conn = sqlite3.connect(DATABASE)
        query = """
            SELECT timestamp, speed, risk_level 
            FROM driving_log 
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(query, conn, params=(int(session_start_time.timestamp()*1000),))
        conn.close()

        if df.empty or len(df) < 2:
            report_text = "Not enough driving data recorded for a report."
        else:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['duration'] = df['timestamp_dt'].diff().dt.total_seconds().fillna(0)
            
            # Calculate statistics
            start_time = df['timestamp_dt'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
            end_time = df['timestamp_dt'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
            total_duration = (df['timestamp_dt'].iloc[-1] - df['timestamp_dt'].iloc[0]).total_seconds()
            avg_speed = df['speed'].mean()
            max_speed = df['speed'].max()
            
            # Behavior analysis
            behavior_summary = df.groupby('risk_level')['duration'].sum()
            total_behavior_time = behavior_summary.sum()
            
            # Generate report
            report_parts = [
                "=== DRIVING BEHAVIOR ANALYSIS REPORT ===",
                f"\nDriver: {driver_name}",
                f"Session Start: {start_time}",
                f"Session End: {end_time}",
                f"Duration: {total_duration:.2f} seconds",
                f"\n--- Speed Statistics ---",
                f"Average Speed: {avg_speed:.2f} km/h",
                f"Maximum Speed: {max_speed:.2f} km/h",
                f"\n--- Behavior Summary ---"
            ]
            
            for behavior, duration in behavior_summary.items():
                percentage = (duration / total_behavior_time * 100) if total_behavior_time > 0 else 0
                report_parts.append(
                    f"{behavior}: {duration:.2f}s ({percentage:.1f}%)"
                )
            
            # Safety score calculation
            aggressive_time = behavior_summary.get('Aggressive', 0)
            safety_score = max(0, 100 - (aggressive_time / total_duration * 100)) if total_duration > 0 else 100
            report_parts.extend([
                f"\n--- Safety Evaluation ---",
                f"Safety Score: {safety_score:.1f}/100",
                f"\n=== END OF REPORT ==="
            ])
            
            report_text = "\n".join(report_parts)

        emit('driving_report', {'report': report_text})
    except Exception as e:
        error_msg = f"Error generating report: {str(e)}"
        print(f"ERROR: {error_msg}")
        emit('driving_report', {'report': error_msg})

# --- Main Execution Block ---
if __name__ == '__main__':
    init_db()
    try:
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nINFO: Shutting down server...")
        if simulator_process and simulator_process.poll() is None:
            simulator_process.terminate()
        sys.exit(0)