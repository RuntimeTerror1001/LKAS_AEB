# Lane-Keeping & Automatic Emergency Braking (LKAS-AEB) System

## Overview
This project implements an advanced ADAS (Advanced Driver Assistance System) featuring Lane Keeping Assist (LKAS) and Automatic Emergency Braking (AEB) capabilities. Built on the CARLA simulator with ROS2 integration, the system provides:

- **Lane Detection & Tracking**: Computer vision-based lane detection using HSV filtering and Hough transforms  
- **Pure Pursuit Control**: Adaptive path-following algorithm for lane keeping  
- **Obstacle Detection**: YOLOv8-based object detection with multi-object tracking  
- **Collision Prediction**: Time-to-collision (TTC) calculation with multi-stage braking  
- **Emergency Braking**: Distance-based and TTC-based braking strategies  

## Key Features

### Perception System
- **Lane Detection**: Real-time lane detection with perspective transformation and polynomial fitting  
- **Obstacle Detection**: YOLOv8-based detection with class-specific distance estimation  
- **Multi-Object Tracking**: Kalman-filter based tracking for consistent obstacle identification  

### Control System
- **Adaptive Pure Pursuit**: Speed-dependent lookahead distance for smooth steering  
- **Speed PID Controller**: Adaptive gains with anti-windup and dead zone handling  
- **Multi-Stage AEB**: Warning, critical, and emergency braking thresholds  
- **Velocity Estimation**: Relative speed calculation using tracking history  

### Visualization
- **RViz Integration**: Real-time visualization of sensor data and control commands  
- **Bird’s-Eye View**: Lane visualization from a top-down perspective  
- **Obstacle Tracking**: Visual markers for tracked objects  
- **Trajectory Markers**: Visualization of planned and executed vehicle paths  

## System Architecture

### RQT Graph
![System Achitecture Diagram](docs/rosgraph.png)

## Prerequisites
- Ubuntu 22.04  
- ROS 2 Humble  
- CARLA 0.9.15  
- Python 3.8+  
- NVIDIA GPU (recommended)  

## Installation
1. **Set up CARLA and ROS2 Bridge**  
   ```bash
   # Install CARLA
   sudo apt install carla-simulator

   # Install ROS2 Bridge
   sudo apt install ros-humble-carla-ros-bridge

2. **Create workspace and clone repository**  
   ```bash
    mkdir -p ~/lkas_aeb_ws/src
    cd ~/lkas_aeb_ws/src
    git clone https://github.com/RuntimeTerror/lkas_aeb.git

3. **Install dependencies**  
   ```bash
   cd ~/lkas_aeb_ws
   rosdep install --from-paths src --ignore-src -r -y

4. **Build workspace**  
   ```bash
    colcon build --symlink-install
    source install/setup.bash

## Usage
1. **Launch CARLA Simulator**  
   ```bash
   ./CarlaUE4.sh -quality-level=Epic

2. **Launch ADAS System**  
   ```bash
    ros2 launch lkas_aeb adas.launch.py

## Configuration
The system is highly configurable through YAML parameter files located in config/params/:
- **Lane Detection Parameters**: lkas_params.yaml
- **AEB Parameters**: aeb_params.yaml

Key adjustable parameters include:
- Confidence thresholds
- Braking force limits
- TTC thresholds
- PID controller gains
- Region of interest settings

## Custom Messages
- **LaneInfo.msg:** Lane center, curvature, and confidence
- **Obstacle.msg:** Object class, distance, speed, and bounding box
- **ObstacleArray.msg:** Collection of detected obstacles
- **TTC.msg:** Time-to-collision and criticality data