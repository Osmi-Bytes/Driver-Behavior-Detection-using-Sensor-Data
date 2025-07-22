# NeuroDrive: Real-Time Driver Behavior Analysis

NeuroDrive is a proof-of-concept system for real-time driver behavior analysis using machine learning. It addresses the high cost and complexity of traditional hardware-based monitoring by providing a complete, software-based pipeline. The system uses a vehicle simulator to generate telemetry data, which is then classified by a machine learning model and visualized on a live web dashboard.

---

## Core Features

- **Real-time Vehicle Simulator:** A GUI built with Tkinter that simulates vehicle movement and generates live IMU data (accelerometer & gyroscope).
- **Live Data Streaming:** Utilizes Flask-SocketIO to transmit sensor data from the simulator to the backend server with low latency.
- **Machine Learning Classification:** A pre-trained TensorFlow/Keras model classifies driving behavior into `Normal`, `Aggressive`, or `Slow` categories.
- **Web-Based Monitoring Dashboard:** A live dashboard that displays real-time driving telemetry and the classified behavior.
- **On-Demand Session Reporting:** Generates a detailed statistical report for the current driving session, including timestamps and a summary of behaviors.

---

## Technology Stack

### Backend
- **Framework:** Flask
- **Real-time Communication:** Flask-SocketIO
- **Asynchronous Server:** Eventlet
- **Machine Learning:** TensorFlow, Keras

### Simulator
- **Language:** Python
- **GUI:** Tkinter

### Frontend
- **Structure & Logic:** HTML, CSS, JavaScript
- **Real-time Communication:** Socket.IO Client JS

### Database & Data Processing
- **Database:** SQLite3
- **Libraries:** Pandas, NumPy

---

## System Architecture

The system operates on a decoupled, three-component architecture:

1.  **Simulator (`simulator.py`):** The data source. An operator drives the virtual car, and the application generates sensor data mimicking a smartphone's IMU, which is then streamed to the server.
2.  **Server (`app.py`):** The central processing core. It ingests the data stream, uses the ML model for inference, stores the results in the database, and broadcasts live updates to all connected clients.
3.  **Dashboard (`dashboard.html`):** The user interface for monitoring. It receives the broadcasted data from the server and displays it live in a web browser.

---

## Setup and Installation

### Prerequisites

- **Python 3.9** is strongly recommended due to TensorFlow dependencies.
- `pip` package manager.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd NeuroDrive
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialize the database:**
    This script creates the `driving_behavior.db` file required for logging.
    ```bash
    python init_db.py
    ```
---

## How to Run

To launch the full system, you will need two separate terminal sessions.

1.  **Start the Backend Server:**
    In your first terminal, execute the following command:
    ```bash
    python app.py
    ```
    The server will start and listen for connections on `http://localhost:5000`.

2.  **Run the Vehicle Simulator:**
    In a new terminal, run the simulator script:
    ```bash
    python simulator.py
    ```
    A GUI window will appear and should confirm its connection to the server.

3.  **Open the Dashboard:**
    Navigate to `http://localhost:5000` in your web browser to view the live dashboard.

---

## Future Scope

- **Mobile Application Development:** Build a standalone mobile application to capture sensor data from a real vehicle, replacing the simulator.
- **Sensor Fusion Implementation:** Integrate GPS and OBD-II data streams to enable the detection of contextual risks, such as speeding.
- **Fleet Management Expansion:** Develop the system into a full-scale platform for commercial fleet management with a centralized dashboard for monitoring multiple drivers.
- **Advanced Model Development:** Collect a larger, more diverse real-world dataset to train more advanced deep learning models for higher accuracy and generalization.

---

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request to suggest improvements or add features.

---

## Author

- **Osama Malik** - [osamamaliktoru@gmail.com](mailto:osamamaliktoru@gmail.com)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.