# Inertial Navigation System (INS)

This project implements an Inertial Navigation System in Python. The implementation covers the fundamental steps required to calculate a trajectory based on raw IMU (Inertial Measurement Unit) data.

## Features
- **Euler Initial Alignment**: Uses static IMU data to estimate initial attitude (roll, pitch, and yaw) before movement.
- **Strapdown Mechanization**: Processes continuous moving IMU data sequentially to dynamically compute position, velocity, and attitude.
- **Performance Analysis**: Estimates positioning and velocity error growth over time using a Van Loan covariance propagation (Kalman Filter error state space).
- **Transportation Rate Analysis**: Computes the translation effects over the Earth's surface relative to the geographic frame (NED).

## Dataset
The code expects an included dataset (`.mat` format) which contains raw measurements (time, specific force, and angular velocity) from an IMU sensor (iIMU-FSAS).

## Requirements
- `numpy`
- `scipy`
- `matplotlib`

## Usage
Run the main script directly through Python:
```bash
python Ex3_IN.py
```
