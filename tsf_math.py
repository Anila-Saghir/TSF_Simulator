# tsf_math.py
import numpy as np
import pandas as pd

# ---------- Rotation matrix (ZYX: yaw, pitch, roll) ----------
def rotation_matrix(roll, pitch, yaw):
    """Euler angles in degrees -> rotation matrix (Rz * Ry * Rx)."""
    r, p, y = np.radians([roll, pitch, yaw])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,         1]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0,          0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    return Rz @ Ry @ Rx

# ---------- Ring anchor generation ----------
def ring_points(radius, z=0.0, phase_deg=0.0):
    """
    Generate 6 anchor points placed evenly around a circle (hexagon pattern).
    radius: ring radius (mm)
    z: z-offset for ring plane (mm)
    phase_deg: angular phase shift for the ring (degrees)
    returns: (6,3) numpy array
    """
    base_angles = np.array([0, 60, 120, 180, 240, 300], dtype=float) + phase_deg
    angs = np.deg2rad(base_angles)
    x = radius * np.cos(angs)
    y = radius * np.sin(angs)
    zv = np.full(6, z, dtype=float)
    pts = np.stack([x, y, zv], axis=1)
    return pts

# ---------- Core inverse kinematics ----------
def compute_leg_lengths(Bi, Pi, T_vec, roll, pitch, yaw):
    """
    Compute leg lengths L_i = || T + R*P_i - B_i || for i=1..6
    Bi: (6,3) base anchor points (in base frame)
    Pi: (6,3) platform anchor points (in platform local frame)
    T_vec: (3,) translation vector for platform origin in base frame
    roll,pitch,yaw: degrees
    returns: numpy array shape (6,) of lengths (mm)
    """
    R = rotation_matrix(roll, pitch, yaw)
    L = np.zeros(6)
    for i in range(6):
        top = T_vec + R.dot(Pi[i])
        L[i] = np.linalg.norm(top - Bi[i])
    return L

# ---------- Apply mounting offsets ----------
def apply_mounting_offsets(Pi, ap_offset=0.0, ap_dir='Medial to Origin',
                           lat_offset=0.0, lat_dir='Anterior to Origin',
                           axial_offset=0.0, axial_dir='Proximal to Origin'):
    """
    Apply mounting offsets to the platform anchor coordinates Pi.
    Offsets meaning: shift the platform center relative to base/bone origin.
    Conventions:
      - AP offset: positive moves platform in +X if 'Medial to Origin', else -X
      - Lateral (here mapped to +Y) offset: 'Anterior to Origin' -> +Y, 'Posterior' -> -Y
      - Axial offset: 'Proximal to Origin' -> +Z, 'Distal to Origin' -> -Z
    Note: coordinate axes are user/visual conventions — adjust if needed.
    Returns shifted Pi (new array).
    """
    # Choose signs per direction strings (these match UI labels in your app)
    ap_sign = 1.0 if ap_dir.lower().startswith('medial') else -1.0
    lat_sign = 1.0 if lat_dir.lower().startswith('anterior') else -1.0
    axial_sign = 1.0 if axial_dir.lower().startswith('proximal') else -1.0

    # Create a translation vector applied to each Pi
    t = np.array([ap_sign * ap_offset, lat_sign * lat_offset, axial_sign * axial_offset], dtype=float)
    return Pi + t  # broadcast addition

# ---------- Plan correction schedule & compute struts ----------
def compute_strut_schedule(start_pose, target_pose, Bi, Pi,
                           days=10):
    """
    Compute per-day poses (linear interpolation), strut lengths each day,
    and the daily delta adjustments.

    start_pose, target_pose: dicts with keys 'tx','ty','tz','roll','pitch','yaw'
    Bi, Pi: anchor points (6x3 arrays)
    days: integer #days (N) -> return N rows from day0..dayN-1

    Returns:
      daily_plan_df: pandas DataFrame with columns ['tx','ty','tz','roll','pitch','yaw']
      strut_df: DataFrame with columns ['Strut 1',...,'Strut 6']
      delta_df: DataFrame with same columns representing per-day change (from previous day).
                For day0 deltas are 0.0.
    """
    # Build daily poses using linear interpolation
    keys = ['tx','ty','tz','roll','pitch','yaw']
    grid = {k: np.linspace(float(start_pose.get(k,0.0)), float(target_pose.get(k,0.0)), days) for k in keys}
    daily_plan_df = pd.DataFrame(grid)

    # Compute strut lengths for each day
    strut_records = []
    for idx, row in daily_plan_df.iterrows():
        Tvec = np.array([row['tx'], row['ty'], row['tz']], dtype=float)
        Ls = compute_leg_lengths(Bi, Pi, Tvec, row['roll'], row['pitch'], row['yaw'])
        strut_records.append(Ls.tolist())
    strut_df = pd.DataFrame(strut_records, columns=[f"Strut {i}" for i in range(1,7)])

    # Compute daily deltas: change compared to previous day (day0 -> zeros)
    delta_records = []
    prev = None
    for idx in range(len(strut_df)):
        if idx == 0:
            delta_records.append([0.0]*6)
            prev = strut_df.iloc[0].values
        else:
            current = strut_df.iloc[idx].values
            delta = (current - prev).tolist()  # positive => lengthening
            delta_records.append(delta)
            prev = current
    delta_df = pd.DataFrame(delta_records, columns=[f"Δ Strut {i}" for i in range(1,7)])

    return daily_plan_df, strut_df, delta_df
