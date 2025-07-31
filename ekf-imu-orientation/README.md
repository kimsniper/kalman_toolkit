# Extended Kalman Filter Demo

This project demonstrates how the **Kalman Filter** can be used to improve orientation estimation (roll, pitch, and yaw) using data from a 6-DoF IMU (accelerometer + gyroscope). It also compares the results with a **Complementary Filter** to show the benefits of Kalman-based sensor fusion.  

The project includes a **real-time Python UI** to visualize raw IMU data, Complementary Filter outputs, and Kalman Filter outputs side-by-side.  

---

## Features
- Real-time orientation estimation (roll, pitch, and yaw)  
- **Kalman Filter** implementation using the [Eigen](https://eigen.tuxfamily.org/) library for clean linear algebra  
- **Complementary Filter** implementation for comparison  
- Real-time Python UI visualization of:  
  - Raw sensor data  
  - Complementary Filter output  
  - Kalman Filter output  

> **Note:** This demo does not use a magnetometer. Yaw estimation is purely based on the gyroscope, so **it will drift over time**.  

---

## Hardware
- MCU: ESP32 (or any similar microcontroller)
- IMU: MPU6050 or any 6-DoF accelerometer + gyroscope module
- USB connection to stream IMU data to PC for visualization

---

## Software
### Firmware
- Implemented in **C++** using ESP-IDF (or PlatformIO)
- Main entry point:  
  [`ekf-imu-orientation/main/main.cpp`](https://github.com/kimsniper/kalman_toolkit/blob/main/ekf-imu-orientation/main/main.cpp)  
- Modular components:  
  - `components/mpu6050/` – IMU driver (I2C)  
  - `components/Eigen/` – Eigen linear algebra library used for Kalman Filter math  

### Host (PC)
- **Python 3** visualization script:  
  - `plot_data.py` reads serial data from the microcontroller  
  - Displays side-by-side plots (raw data vs. filters) in real-time  

---

## Repository Structure

```
kalman_toolkit/
├── ekf-imu-orientation/
│ ├── main/
│ │ └── main.cpp # Firmware entry point
│ ├── components/
│ │ ├── Eigen/ # Eigen linear algebra library
│ │ └── mpu6050/ # IMU driver
│ ├── CMakeLists.txt
│ └── Kconfig.projbuild
├── host/
│ ├── plot_data.py
│ ├── requirements.txt
│ └── ui/ # UI modules for Python visualization
├── LICENSE.txt
├── README.md
└── .gitignore
```

---

## Demo
The real-time Python UI displays:
1. **Complementary Filter orientation estimate**  
2. **Kalman Filter orientation estimate**  

This makes it easy to observe:
- Accelerometer noise  
- Gyroscope drift  
- How sensor fusion improves stability  

---

## Installation
### Firmware
1. Clone the repository:  
```bash
git clone https://github.com/kimsniper/kalman_toolkit.git
cd kalman_toolkit/ekf-imu-orientation
```

2. Build & flash
```bash
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

3. Verify
```bash
idf.py --version  # Project is based on v5.4.2
```

---

## Data Visualization UI (`plot_data.py`)

### Requirements

Install dependencies:

```bash
pip install matplotlib pyserial
```

### Run the UI

```bash
python3 plot_data.py
```

---

## Video Demo

Link: https://www.linkedin.com/posts/mezaeldocoy_kalmanfilter-imu-sensorfusion-activity-7356550992500281345-WmCh?utm_source=share&utm_medium=member_android&rcm=ACoAACmY-xYBsymlj36REm4IhJ-hJ5gTkK0J9l0
