import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.linalg import expm

# ============================================================
# EX03 - Full Script (Euler Initial Alignment + Part 2 Mechanization)
# Conventions:
#   - n-frame: NED = [North, East, Down]
#   - IMU columns: [t, wx, wy, wz, fx, fy, fz]
#   - f^b is specific force [m/s^2]
# ============================================================

# ----------------Load .mat file (EX03 dataset)
mat_file_path = "ex03_10075939.mat"
mat_data = scipy.io.loadmat(mat_file_path)

static_data = mat_data["imudata"]["static"][0, 0]   # (Ns, 7)
moving_data = mat_data["imudata"]["moving"][0, 0]   # (Nm, 7)

print("static_data shape:", static_data.shape)
print("moving_data shape:", moving_data.shape)

# ============================================================
# Part 1: Euler Initial Alignment (ONLY Euler method)
# ============================================================

# ------------Split static block
Time_static = static_data[:, 0]       # [s]
omega_bx_s  = static_data[:, 1]       # [rad/s]
omega_by_s  = static_data[:, 2]
omega_bz_s  = static_data[:, 3]
f_bx_s      = static_data[:, 4]       # [m/s^2] specific force
f_by_s      = static_data[:, 5]
f_bz_s      = static_data[:, 6]

dt_static = float(np.mean(np.diff(Time_static)))
print(f"Static dt = {dt_static:.6f} s  (~{1/dt_static:.1f} Hz)")

# -----------Static mean vectors
f_b_mean = np.array([f_bx_s.mean(), f_by_s.mean(), f_bz_s.mean()])
omega_ib_b = np.array([omega_bx_s.mean(), omega_by_s.mean(), omega_bz_s.mean()])

# -----------Gravity magnitude (from specific force mean magnitude)
g = float(np.linalg.norm(f_b_mean))

# -----------Earth rotation magnitude estimate (report only)
omega_mag_rad_s = float(np.linalg.norm(omega_ib_b))
omega_mag_deg_h = np.rad2deg(omega_mag_rad_s) * 3600.0

print("\nStatic means:")
print("f_b_mean [m/s^2] =", f_b_mean)
print("omega_ib_b [rad/s] =", omega_ib_b)
print(f"Estimated g = {g:.6f} m/s^2")
print(f"Estimated |omega| = {omega_mag_deg_h:.6f} deg/h")

# -----------Initial position/velocity (Task 1a)
x0_n = np.array([0.0, 0.0, 0.0])
v0_n = np.array([0.0, 0.0, 0.0])

# ------------------------------------------------------------
# Euler rotation matrices (same as you used)
def C1(phi):
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi), np.sin(phi)],
        [0.0, -np.sin(phi), np.cos(phi)]
    ])

def C2(theta):
    return np.array([
        [np.cos(theta), 0.0, -np.sin(theta)],
        [0.0, 1.0, 0.0],
        [np.sin(theta), 0.0, np.cos(theta)]
    ])

def C3(psi):
    return np.array([
        [np.cos(psi), np.sin(psi), 0.0],
        [-np.sin(psi), np.cos(psi), 0.0],
        [0.0, 0.0, 1.0]
    ])

# ------------------------------------------------------------
# Euler initial alignment ONLY (roll, pitch from accel; yaw from leveled gyro)
fx, fy, fz = f_b_mean

# You insisted on this sign convention (keep it consistent with your earlier work)
roll_0  = np.arctan2(-fy, -fz)
pitch_0 = np.arctan(fx / np.sqrt(fy**2 + fz**2))

w_ib_n_prime = C2(pitch_0).T @ (C1(roll_0).T @ omega_ib_b)
yaw_0 = np.arctan2(-w_ib_n_prime[1], w_ib_n_prime[0])

# Slides define C_b^n = C1*C2*C3, and C_n^b = (C_b^n)^T
C_b_n_0 = C1(roll_0) @ C2(pitch_0) @ C3(yaw_0)
C_n_b_0 = C_b_n_0.T

print("\nInitial Euler [Roll, Pitch, Yaw] =", np.rad2deg([roll_0, pitch_0, yaw_0]), "deg")
print("Initial DCM C_n_b =")
print(np.round(C_n_b_0, 7))

# ------------------------------------------------------------
# Latitude estimate from Earth-rate direction in n-frame (using C_n^b)
def normalize(v, eps=1e-15):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

w_b_dir = normalize(omega_ib_b)
w_n_dir = normalize(C_n_b_0 @ w_b_dir)

# NED earth-rate direction: [cos(lat), 0, -sin(lat)]
lat_rad_est = np.arctan2(-w_n_dir[2], w_n_dir[0])

# Wrap to [-90°, +90°]
if lat_rad_est > np.pi/2:
    lat_rad_est = np.pi - lat_rad_est
elif lat_rad_est < -np.pi/2:
    lat_rad_est = -np.pi - lat_rad_est

lat_deg_est = np.rad2deg(lat_rad_est)
print(f"Estimated latitude = {lat_deg_est:.6f} deg")

# ============================================================
# Part 2: Mechanization (use moving block)
# ============================================================

# ------------Split moving block
Time_data = moving_data[:, 0]
omega_bx  = moving_data[:, 1]   # [rad/s]
omega_by  = moving_data[:, 2]
omega_bz  = moving_data[:, 3]
f_bx      = moving_data[:, 4]   # [m/s^2]
f_by      = moving_data[:, 5]
f_bz      = moving_data[:, 6]

dt_s = float(np.mean(np.diff(Time_data)))
elapsed_time = Time_data - Time_data[0]
print(f"\nMoving dt = {dt_s:.6f} s, duration = {elapsed_time[-1]:.3f} s")

# Optional: enforce motion duration ~52 s (recommended if file contains extra)
# T_move = 52.0
# mask = elapsed_time <= T_move
# elapsed_time = elapsed_time[mask]
# omega_bx, omega_by, omega_bz = omega_bx[mask], omega_by[mask], omega_bz[mask]
# f_bx, f_by, f_bz = f_bx[mask], f_by[mask], f_bz[mask]
# dt_s = float(np.mean(np.diff(elapsed_time)))

# ----------------------------
# Earth rotation vector in n-frame (NED)
siderial_day_sec = 86164.09
Omega_e_rad_s = (2.0 * np.pi) / siderial_day_sec
omega_ie_n = Omega_e_rad_s * np.array([np.cos(lat_rad_est), 0.0, -np.sin(lat_rad_est)])

# Gravity acceleration (NED, Down positive)
g_n = np.array([0.0, 0.0, g])

# ----------------------------
# Helpers for attitude update (Rodrigues / expm)
# This is NOT an “extra initial alignment solution”.
# It’s the standard discrete attitude update needed for Part 2.
def skew(a):
    return np.array([
        [0.0,   -a[2],  a[1]],
        [a[2],   0.0,  -a[0]],
        [-a[1],  a[0],  0.0]
    ])

def reorthogonalize(C):
    U, _, Vt = np.linalg.svd(C)
    return U @ Vt

def euler_from_C_n_b(C):
    # Your corrected index set for C_n^b
    roll  = np.arctan2(C[1,2], C[2,2])
    pitch = np.arcsin(-C[0,2])
    yaw   = np.arctan2(C[0,1], C[0,0])
    return roll, pitch, yaw

# ----------------------------
# Initialize states
C_n_b = C_n_b_0.copy()
v_n = v0_n.copy()
x_n = x0_n.copy()

# Storage for plots
roll_series  = []
pitch_series = []
yaw_series   = []
speed_series = []
pos_series   = []
v_series     = []
# ----------------------------
# Main strapdown loop
for k in range(len(elapsed_time)):
    omega_ib_b_k = np.array([omega_bx[k], omega_by[k], omega_bz[k]])
    f_b_k        = np.array([f_bx[k], f_by[k], f_bz[k]])

    # --------------------------------------------------------
    # Attitude update:
    # omega_nb^b = omega_ib^b - omega_ie^b
    # omega_ie^b = C_b^n * omega_ie^n = (C_n^b)^T * omega_ie^n
    omega_ie_b = C_n_b.T @ omega_ie_n
    omega_nb_b = omega_ib_b_k - omega_ie_b

    # Discrete rotation: C_{k+1} = C_k * exp([delta_theta×])
    delta_theta = omega_nb_b * dt_s
    C_n_b = C_n_b @ expm(skew(delta_theta))

    # Re-orthogonalize to keep C in SO(3) (prevents drift)
    C_n_b = reorthogonalize(C_n_b)

    # --------------------------------------------------------
    # Velocity update:
    # v_dot^n = C_n^b f^b + g^n - 2*omega_ie^n × v^n  (transport omitted, handled later in Task 4)
    f_n = C_n_b @ f_b_k
    coriolis = 2.0 * np.cross(omega_ie_n, v_n)
    v_dot_n = f_n + g_n - coriolis
    v_next = v_n + v_dot_n * dt_s

    # --------------------------------------------------------
    # Position update (tangential plane): trapezoid integration
    x_next = x_n + 0.5 * (v_n + v_next) * dt_s

    v_n = v_next
    x_n = x_next
    v_series.append(v_n.copy())
    # Save series
    roll, pitch, yaw = euler_from_C_n_b(C_n_b)
    roll_series.append(np.rad2deg(roll))
    pitch_series.append(np.rad2deg(pitch))
    yaw_series.append(np.rad2deg(yaw))
    speed_series.append(np.linalg.norm(v_n))
    pos_series.append(x_n.copy())

# ----------------------------
# Final friend-style outputs
print("\nFinal Position [N, E, D] =", np.round(x_n, 4), "m")
print("Final Velocity [N, E, D] =", np.round(v_n, 4), "m/s")
print("Final Euler [Roll, Pitch, Yaw] =", np.round([roll_series[-1], pitch_series[-1], yaw_series[-1]], 4), "deg")
print("Final DCM C_n_b =")
print(np.round(C_n_b, 7))

# ============================================================
# Required plots (Task 2b)
# ============================================================

def format_y_tick(value, tick_number):
    return f"{value:.2f}"

# 1) Euler angles vs time
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(elapsed_time, roll_series, linewidth=2)
axes[0].set_ylabel("Roll [deg]", fontweight="bold", fontsize=12)
axes[0].grid(True)

axes[1].plot(elapsed_time, pitch_series, linewidth=2)
axes[1].set_ylabel("Pitch [deg]", fontweight="bold", fontsize=12)
axes[1].grid(True)

axes[2].plot(elapsed_time, yaw_series, linewidth=2)
axes[2].set_ylabel("Yaw [deg]", fontweight="bold", fontsize=12)
axes[2].set_xlabel("Time [s]", fontweight="bold", fontsize=12)
axes[2].grid(True)

for ax in axes:
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_y_tick))
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight("bold")
        label.set_fontsize(9)

plt.tight_layout()
plt.show()

# 2) Speed magnitude vs time
plt.figure(figsize=(12, 4.5))
plt.plot(elapsed_time, speed_series, linewidth=2)
plt.grid(True)
plt.xlabel("Time [s]", fontweight="bold", fontsize=12)
plt.ylabel("|v| [m/s]", fontweight="bold", fontsize=12)

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_y_tick))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight("bold")
    label.set_fontsize(9)

plt.tight_layout()
plt.show()

# 3) Trajectory on tangential plane (n-frame): East vs North
pos_arr = np.asarray(pos_series)
N_arr = pos_arr[:, 0]
E_arr = pos_arr[:, 1]

plt.figure(figsize=(6.5, 6.5))
plt.plot(E_arr, N_arr, linewidth=2)
plt.grid(True)
plt.xlabel("East [m]", fontweight="bold", fontsize=12)
plt.ylabel("North [m]", fontweight="bold", fontsize=12)

ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight("bold")
    label.set_fontsize(9)

plt.tight_layout()
plt.show()


#---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# ============================================================
# EX03 - Part 3a: Performance Analysis (Van Loan)
# Error state: δx = [δx, δv, δψ]  (each 3x1)  -> total 9x1
#
# Given (exercise discussion):
# [δx_dot]   [0   I   0] [δx]   [0 0] [δf]
# [δv_dot] = [0 -2Ω  -[f×]] [δv] + [I 0] [δω]
# [δψ_dot]   [0   0   0] [δψ]   [0 I]
#
# P_{k+1} = Φ P_k Φ^T + Q_k, with (Φ, Q_k) from Van Loan method.
# ============================================================

# -----------------------------
# Inputs you already have from Part 1/2:
#   dt_s        : sampling interval of the MOVING block (seconds)
#   lat_rad_est : latitude estimate in radians
#   g           : local gravity magnitude (m/s^2)
#
# If you don't have them in this cell, define them here or pass them in.
# -----------------------------

# Example placeholders (REMOVE these if you already have them)
# dt_s = 0.01
# lat_rad_est = np.deg2rad(52.385828)
# g = 9.81

# -----------------------------
# Helper: skew-symmetric matrix [a×]
def skew(a):
    return np.array([
        [0.0,   -a[2],  a[1]],
        [a[2],   0.0,  -a[0]],
        [-a[1],  a[0],  0.0]
    ])

# -----------------------------
# Van Loan discretization (same as provided MATLAB function)
def vanLoan(A, B, Qu, dt):
    """
    Based on Brown/Hwang and the Ex03 discussion slide:
    M = [ -A dt    B Qu B^T dt
           0       A^T dt  ]
    N = expm(M)
    Φ = N(10:18,10:18)^T
    Q = Φ * N(1:9,10:18)
    """
    M = np.block([
        [-A*dt,           B @ Qu @ B.T * dt],
        [np.zeros((9,9)),  A.T * dt]
    ])
    N = expm(M)
    PHI = N[9:18, 9:18].T
    Qk  = PHI @ N[0:9, 9:18]
    return PHI, Qk

# ============================================================
# 1) Build A and B (from the Ex03 discussion)
# ============================================================

# Earth rotation in n-frame (NED):
# ω_ie^n = Ω_E [cos(lat), 0, -sin(lat)]^T
siderial_day_sec = 86164.09
Omega_E = 2.0 * np.pi / siderial_day_sec
omega_ie_n = Omega_E * np.array([np.cos(lat_rad_est), 0.0, -np.sin(lat_rad_est)])

Omega_ie = skew(omega_ie_n)

# Constant specific force "only gravity" in n-frame.
# In NED, gravity acceleration is +Down. But specific force at rest is ~ -g in Down.
# Exercise says assume constant specific forces (only gravity). :contentReference[oaicite:8]{index=8}
f_n_const = np.array([0.0, 0.0, -g])
F = skew(f_n_const)

Z = np.zeros((3,3))
I = np.eye(3)

# A matrix (9x9), exactly as in the slide (Eq. 7 there) :contentReference[oaicite:9]{index=9}
A = np.block([
    [Z,              I,              Z],
    [Z,   -2.0*Omega_ie,           -F],
    [Z,              Z,              Z]
])

# B matrix (9x6), exactly as in the slide :contentReference[oaicite:10]{index=10}
B = np.block([
    [Z, Z],
    [I, Z],
    [Z, I]
])

# ============================================================
# 2) Build Qu (6x6) from iMAR iIMU-FSAS datasheet white noise
# ============================================================

# Standard gravity constant for unit conversion from "g" to m/s^2
g0 = 9.80665

# From your iIMU-FSAS datasheet:
# - Angular random walk: 0.15 °/√h
# - Acc noise density: < 50 µg/√Hz  (datasheet prints "µg/Hz", but it is noise density)
# :contentReference[oaicite:11]{index=11}
ARW_deg_sqrt_h = 0.15
acc_noise_ug_sqrt_Hz = 50.0

# Convert gyro ARW -> rad/√s (divide by 60 because √3600 = 60)
sigma_w = (ARW_deg_sqrt_h / 60.0) * (np.pi / 180.0)

# Convert accel noise density -> m/s^2/√Hz
sigma_f = (acc_noise_ug_sqrt_Hz * 1e-6) * g0

# Qu: PSD on diagonal (accelerometers then gyros), as stated in discussion :contentReference[oaicite:12]{index=12}
Qu = np.diag([
    sigma_f**2, sigma_f**2, sigma_f**2,
    sigma_w**2, sigma_w**2, sigma_w**2
])

print("sigma_f [m/s^2/√Hz] =", sigma_f)
print("sigma_w [rad/√s]    =", sigma_w)

# ============================================================
# 3) Initial covariance P0
# ============================================================

# Initial position and velocity error-free.
# Orientation uncertainty: 1 mrad on all axes. :contentReference[oaicite:13]{index=13}
sig_psi0 = 1e-3  # rad

P = np.zeros((9,9))
P[6,6] = sig_psi0**2
P[7,7] = sig_psi0**2
P[8,8] = sig_psi0**2

# ============================================================
# 4) Discretize (Van Loan) and propagate to t=52 s
# ============================================================

PHI, Qk = vanLoan(A, B, Qu, dt_s)

T_end = 52.0
Nsteps = int(np.floor(T_end / dt_s)) + 1
t = np.arange(Nsteps) * dt_s

sigma_pos = np.zeros(Nsteps)  # magnitude of position std vector
sigma_vel = np.zeros(Nsteps)  # magnitude of velocity std vector

for k in range(Nsteps):
    # magnitude of std vectors (as requested: "Plot magnitude of the vectors") :contentReference[oaicite:14]{index=14}
    sigma_pos[k] = np.sqrt(P[0,0] + P[1,1] + P[2,2])
    sigma_vel[k] = np.sqrt(P[3,3] + P[4,4] + P[5,5])

    P = PHI @ P @ PHI.T + Qk

print("\n--- Part 3a results at t=52 s ---")
print("||σ_pos|| [m]   =", sigma_pos[-1])
print("||σ_vel|| [m/s] =", sigma_vel[-1])

# ============================================================
# 5) Plot the magnitudes over time (required)
# ============================================================

plt.figure(figsize=(11,4.5))
plt.plot(t, sigma_vel, linewidth=2)
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("||σ_v|| [m/s]")
plt.tight_layout()
plt.show()

plt.figure(figsize=(11,4.5))
plt.plot(t, sigma_pos, linewidth=2)
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("||σ_x|| [m]")
plt.tight_layout()
plt.show()


#---------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Collect velocity history from Part 2
# v_series should be list of [vN, vE, vD]
v_arr = np.asarray(v_series)                 # shape (N, 3)
vN = v_arr[:, 0]
vE = v_arr[:, 1]
# vD = v_arr[:, 2]  # not needed for omega_en_n in this simplified formula

# ----------------------------
# Constants from Ex03 discussion slide
R0 = 6378000.0   # [m]  R0 ≈ RE ≈ RN
h0 = 70.0        # [m]  ellipsoidal height fixed
Re_h = R0 + h0
Rn_h = R0 + h0

phi = lat_rad_est

# ----------------------------
# Transportation rate omega_en^n = [omega_N, omega_E, omega_D] in rad/s (NED)
omega_en_n = np.zeros((len(v_arr), 3))
omega_en_n[:, 0] =  vE / Re_h
omega_en_n[:, 1] = -vN / Rn_h
omega_en_n[:, 2] = -(vE * np.tan(phi)) / Re_h

# Magnitude (optional, useful for your "is it necessary?" statement)
omega_en_mag = np.linalg.norm(omega_en_n, axis=1)

# Earth rate magnitude for comparison (order)
siderial_day_sec = 86164.09
Omega_E = 2.0 * np.pi / siderial_day_sec
omega_ie_n = Omega_E * np.array([np.cos(phi), 0.0, -np.sin(phi)])
omega_ie_mag = np.linalg.norm(omega_ie_n)   # ~7e-5 rad/s

print("Mean |omega_en| [rad/s] =", float(np.mean(omega_en_mag)))
print("Max  |omega_en| [rad/s] =", float(np.max(omega_en_mag)))
print("|omega_ie| [rad/s]      =", float(omega_ie_mag))

# ----------------------------
# Plot all 3 components in ONE diagram (Task 4 requirement)
plt.figure(figsize=(12, 5))
plt.plot(elapsed_time, omega_en_n[:, 0], linewidth=2, label=r'$\omega_{en,N}^n$')
plt.plot(elapsed_time, omega_en_n[:, 1], linewidth=2, label=r'$\omega_{en,E}^n$')
plt.plot(elapsed_time, omega_en_n[:, 2], linewidth=2, label=r'$\omega_{en,D}^n$')
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel(r"Transportation rate $\omega_{en}^n$ [rad/s]")
plt.legend()
plt.tight_layout()
plt.show()
