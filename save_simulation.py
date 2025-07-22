# save_simulation.py
import pandas as pd
import joblib
import time
import csv

# Load model
model = joblib.load('models/rf_model.pkl')

# Load test data
test_df = pd.read_csv('data/test_data.csv')
features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']

# Prepare output CSV
output_file = 'data/simulation_output.csv'
with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp'] + features + ['Predicted_Class'])

# Simulate continuous prediction
for index, row in test_df.iterrows():
    input_data = row[features].values.reshape(1, -1)
    pred_class = model.predict(input_data)[0]
    
    timestamp = row.get('Timestamp', int(time.time() * 1000))  # Fallback to current time

    # Append to output file
    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp] + list(row[features]) + [pred_class])
    
    print(f"Processed @ {timestamp} | Predicted Class: {pred_class}")
    time.sleep(0.1)  # Simulate delay