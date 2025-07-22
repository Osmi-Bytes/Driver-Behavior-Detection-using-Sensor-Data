import sqlite3
import os

# Ensure instance folder exists
os.makedirs('instance', exist_ok=True)

# Connect to SQLite DB (will create file if not exists)
conn = sqlite3.connect('instance/driving_behavior.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS driving_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp BIGINT,
    acc_x FLOAT,
    acc_y FLOAT,
    acc_z FLOAT,
    gyro_x FLOAT,
    gyro_y FLOAT,
    gyro_z FLOAT,
    speed FLOAT,  
    predicted_class INT,
    risk_level TEXT
)
''')

conn.commit()
conn.close()

print("SQLite DB initialized.")