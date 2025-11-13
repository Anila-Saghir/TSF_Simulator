# main.py 
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tsf_math import (
    rotation_matrix,
    ring_points,
    compute_leg_lengths,
    apply_mounting_offsets,
    compute_strut_schedule,
)

# --- Page setup ---
st.set_page_config(layout="wide", page_title="Taylor Spatial Frame Simulator")
st.markdown("""
<style>
/* Tab labels */
button[role="tab"] {
    border: 2px solid #CBC3E3 !important; 
    border-radius: 5px !important; 
    padding: 5px 10px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Taylor Spatial Frame Simulator")

# --- session keys ---
for k in [
    "patient", "deformity", "frame", "mounting",
    "daily_plan", "strut_df", "delta_df",
    "dist_pts_records", "rot_records",
    "selected_day", "home_dist_center", "home_rot"
]:
    if k not in st.session_state:
        st.session_state[k] = None

# ---------- geometry helpers ----------
def cylinder_mesh(p0, p1, radius=3.0, segments=16):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-6: return None
    vz = v / L
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, vz)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    vx = np.cross(vz, arbitrary)
    vx /= np.linalg.norm(vx)
    vy = np.cross(vz, vx)
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    circle0 = [p0 + radius * (np.cos(a) * vx + np.sin(a) * vy) for a in angles]
    circle1 = [p1 + radius * (np.cos(a) * vx + np.sin(a) * vy) for a in angles]
    verts = np.array(circle0 + circle1)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    I, J, K = [], [], []
    n = segments
    for i in range(n):
        i0, i1 = i, (i + 1) % n
        j0, j1 = i + n, i1 + n
        I += [i0, i1]
        J += [j0, j0]
        K += [i1, j1]
    return x, y, z, np.array(I), np.array(J), np.array(K)


def pencil_mesh(p_base, p_tip, base_radius=6.0, tip_radius=2.0, segments=20):
    p0, p1 = np.array(p_base, float), np.array(p_tip, float)
    axis = p1 - p0
    L = np.linalg.norm(axis)
    if L < 1e-6: return None
    zhat = axis / L
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, zhat)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    vx = np.cross(zhat, a)
    vx /= np.linalg.norm(vx)
    vy = np.cross(zhat, vx)
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    base_circle = [p0 + base_radius * (np.cos(t) * vx + np.sin(t) * vy) for t in angles]
    tip_circle = [p1 + tip_radius * (np.cos(t) * vx + np.sin(t) * vy) for t in angles]
    verts = np.array(base_circle + tip_circle)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    I, J, K = [], [], []
    n = segments
    for i in range(n):
        i0, i1 = i, (i + 1) % n
        j0, j1 = i + n, i1 + n
        I += [i0, i1]
        J += [j0, j0]
        K += [i1, j1]
    return x, y, z, np.array(I), np.array(J), np.array(K)


# ---------- UI Tabs ----------
tabs = st.tabs(["Patient Data", "Define Deformities", "Select Frame", "Mounting Frame", "Results"])

# --- Tab 1: Patient Data ---
with tabs[0]:
    with st.form("patient_form"):
        st.markdown('<h2 style="color:red">Patient Data</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col2:
            name = st.text_input("Patient Name", value=st.session_state["patient"].get("name", "") if st.session_state["patient"] else "")
            age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state["patient"].get("age", 18) if st.session_state["patient"] else 18)
            op_code = st.text_input("Operation Code", value=st.session_state["patient"].get("op_code", "") if st.session_state["patient"] else "")
            op_date_input = st.date_input("Operation Date", value=pd.Timestamp.today())

            submitted = st.form_submit_button("Enter Patient Data")
            if submitted:
                if not name.strip():
                    st.error("⚠️ Please enter Patient Name.")
                elif not op_code.strip():
                    st.error("⚠️ Please enter Operation Code.")
                else:
                    st.session_state["patient"] = {"name": name, "age": age, "op_code": op_code, "op_date": str(op_date_input)}
                    st.success("Patient data saved!")

# --- Tab 2: Define Deformities ---
with tabs[1]:
    with st.form("deformity_form"):
        st.markdown('<h2 style="color:red">Define Deformities</h2>', unsafe_allow_html=True)
        
        with st.expander("Angular Measurements"):
            anatomy = st.selectbox("Anatomy", ["Select","Right", "Left"], index=0)
            ref_fragment = st.selectbox("Reference Fragment", ["Select", "Proximal", "Distal"], index=0)
            ap_ang = st.number_input("AP view Angulation (deg)", min_value=0.0, max_value=360.0, value=None, format="%.1f")
            ap_ang_type = st.radio("AP type", ["Valgus", "Varus"], horizontal=True)
            lat_ang = st.number_input("Lateral view Angulation (deg)", min_value=0.0, max_value=360.0, value=None, format="%.1f")
            lat_ang_type = st.radio("Lateral type", ["Apex Posterior", "Apex Anterior"], horizontal=True)
            axial_ang = st.number_input("Axial Angulation (deg)", min_value=0.0, max_value=360.0, value=None, format="%.1f")
            axial_ang_type = st.radio("Axial type", ["External", "Internal"], horizontal=True)
        
        with st.expander("Translational Measurements"):
            ap_trans = st.number_input("AP view Translation (mm)", min_value=0.0, max_value=1000.0, value=None, format="%.1f")
            ap_trans_type = st.radio("AP trans", ["Medial", "Lateral"], horizontal=True)
            lat_trans = st.number_input("Lateral view Translation (mm)", min_value=0.0, max_value=1000.0, value=None, format="%.1f")
            lat_trans_type = st.radio("Lat trans", ["Anterior", "Posterior"], horizontal=True)
            axial_trans = st.number_input("Axial Translation (mm)", min_value=0.0, max_value=1000.0, value=None, format="%.1f")
            axial_trans_type = st.radio("Axial trans", ["Short", "Long"], horizontal=True)
        
        submitted = st.form_submit_button("Enter Deformity Data")
        if submitted:
            missing_fields = [f for f, v in zip(
                ["AP Angulation","Lateral Angulation","Axial Angulation","AP Translation","Lateral Translation","Axial Translation"],
                [ap_ang, lat_ang, axial_ang, ap_trans, lat_trans, axial_trans]
            ) if v is None]
            if missing_fields:
                st.error(f"⚠️ Please complete the following fields: {', '.join(missing_fields)}")
            else:
                st.session_state["deformity"] = {
                    "anatomy": anatomy, "ref_fragment": ref_fragment,
                    "ap_ang": ap_ang, "ap_ang_type": ap_ang_type,
                    "lat_ang": lat_ang, "lat_ang_type": lat_ang_type,
                    "axial_ang": axial_ang, "axial_ang_type": axial_ang_type,
                    "ap_trans": ap_trans, "ap_trans_type": ap_trans_type,
                    "lat_trans": lat_trans, "lat_trans_type": lat_trans_type,
                    "axial_trans": axial_trans, "axial_trans_type": axial_trans_type
                }
                st.success("Deformity data saved!")

# --- Tab 3: Select Frame ---
with tabs[2]:
    with st.form("frame_form"):
        st.markdown('<h2 style="color:red">Select Frame</h2>', unsafe_allow_html=True)
        with st.expander("Ring Dimensions"):
            prox_diam = st.number_input("Proximal Ring Diameter (mm)", min_value=80.0, max_value=300.0, value=None, format="%.1f")
            dist_diam = st.number_input("Distal Ring Diameter (mm)", min_value=80.0, max_value=300.0, value=None, format="%.1f")
        
        with st.expander("Strut Selection"):
            opts = ["Select", "Extra Short 75-96mm", "Short 90-125mm", "Medium 116-178mm", "Long 169-289mm"]
            struts = [st.selectbox(f"Strut {i}", opts, index=0) for i in range(1, 7)]
        
        submitted = st.form_submit_button("Enter Frame Selection")
        if submitted:
            missing_fields = []
            if prox_diam is None: missing_fields.append("Proximal Ring Diameter")
            if dist_diam is None: missing_fields.append("Distal Ring Diameter")
            if any(s=="Select" for s in struts): missing_fields.append("Strut Selection")
            if missing_fields:
                st.error(f"⚠️ Please complete the following fields: {', '.join(missing_fields)}")
            else:
                st.session_state["frame"] = {"prox_diam": prox_diam, "dist_diam": dist_diam, "struts": struts}
                st.success("Frame selection saved!")

# --- Tab 4: Mounting Frame ---
with tabs[3]:
    with st.form("mounting_form"):
        st.markdown('<h2 style="color:red">Mounting Frame</h2>', unsafe_allow_html=True)
        with st.expander("Frame Dimensions"):
            operative_mode = st.selectbox("Operative Mode", ["Select", "Chronic", "Residual"], index=0)
            neutral_height = st.number_input("Neutral Frame Height (mm)", min_value=50.0, max_value=400.0, value=None, format="%.1f")
        with st.expander("Offsets"):
            ap_offset = st.number_input("AP View From Offset (mm)", value=None, format="%.1f")
            ap_offset_dir = st.radio("AP Offset Dir", ["Medial to Origin", "Lateral to Origin"], horizontal=True)
            lat_offset = st.number_input("Lateral View From Offset (mm)", value=None, format="%.1f")
            lat_offset_dir = st.radio("Lat Offset Dir", ["Anterior to Origin", "Posterior to Origin"], horizontal=True)
            axial_offset = st.number_input("Axial Frame Offset (mm)", value=None, format="%.1f")
            axial_offset_dir = st.radio("Axial Offset Dir", ["Proximal to Origin", "Distal to Origin"], horizontal=True)
        
        submitted = st.form_submit_button("Enter Mounting Data")
        if submitted:
            if None in [neutral_height, ap_offset, lat_offset, axial_offset]:
                st.error("⚠️ Please complete all fields before submitting Mounting Data.")
            else:
                st.session_state["mounting"] = {
                    "operative_mode": operative_mode,
                    "neutral_height": float(neutral_height),
                    "ap_offset": float(ap_offset), "ap_offset_dir": ap_offset_dir,
                    "lat_offset": float(lat_offset), "lat_offset_dir": lat_offset_dir,
                    "axial_offset": float(axial_offset), "axial_offset_dir": axial_offset_dir
                }
                st.success("Mounting data saved!")

# --- Tab 5: Results ---
with tabs[4]:
    with st.form("results_form"):
        st.markdown('<h2 style="color:red">Results</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            correction_days = st.number_input("Correction Time (Days)", min_value=2, max_value=60, value=10 )
            run_sim = st.form_submit_button("Run Simulation")

        with col2:
            st.markdown("### Strut Schedule and 3D Visualization appear here after simulation")

        if run_sim:
            if correction_days is None:
                st.error("⚠️ Please enter the Correction Time before running simulation.")
            elif not all([
                st.session_state.get("patient"),
                st.session_state.get("deformity"),
                st.session_state.get("frame"),
                st.session_state.get("mounting")
            ]):
                st.error("⚠️ Please complete all prior tabs before running simulation.")
            else:
                prox_r = st.session_state["frame"]["prox_diam"] / 2.0
                dist_r = st.session_state["frame"]["dist_diam"] / 2.0
                home_z = st.session_state["mounting"]["neutral_height"]

                Bi = ring_points(prox_r, z=0.0, phase_deg=0.0)
                Pi_local = ring_points(dist_r, z=0.0, phase_deg=30.0)
                Pi_local[:, 2] += home_z

                m = st.session_state["mounting"]
                Pi_shifted = apply_mounting_offsets(
                    Pi_local,
                    ap_offset=float(m.get("ap_offset", 0.0)), ap_dir=m.get("ap_offset_dir"),
                    lat_offset=float(m.get("lat_offset", 0.0)), lat_dir=m.get("lat_offset_dir"),
                    axial_offset=float(m.get("axial_offset", 0.0)), axial_dir=m.get("axial_offset_dir"),
                )

                deform = st.session_state["deformity"]
                start_pose = {
                    "tx": float(deform.get("ap_trans", 0.0)),
                    "ty": float(deform.get("lat_trans", 0.0)),
                    "tz": float(deform.get("axial_trans", 0.0)),
                    "roll": float(deform.get("ap_ang", 0.0)),
                    "pitch": float(deform.get("lat_ang", 0.0)),
                    "yaw": float(deform.get("axial_ang", 0.0))
                }
                target_pose = {"tx": 0.0, "ty": 0.0, "tz": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

                daily_plan, strut_df, delta_df = compute_strut_schedule(
                    start_pose, target_pose, Bi, Pi_shifted, days=int(correction_days)
                )

                st.session_state["daily_plan"] = daily_plan
                st.session_state["strut_df"] = strut_df
                st.session_state["delta_df"] = delta_df

                dist_pts_records, rot_records = [], []
                for _, row in daily_plan.iterrows():
                    T = np.array([row["tx"], row["ty"], row["tz"]])
                    R = rotation_matrix(row["roll"], row["pitch"], row["yaw"])
                    pts = np.array([T + R.dot(Pi_shifted[i]) for i in range(6)])
                    dist_pts_records.append(pts)
                    rot_records.append(R)

                st.session_state["dist_pts_records"] = dist_pts_records
                st.session_state["rot_records"] = rot_records
                st.session_state["home_dist_center"] = dist_pts_records[-1].mean(axis=0)
                st.session_state["home_rot"] = rot_records[-1]
                st.session_state["selected_day"] = 0
                st.success("Simulation computed — click a Day to view day by day bone alignment.")

    # ---- Visualization after simulation ----
    if st.session_state["strut_df"] is not None:
        st.subheader("Strut Lengths (mm)")
        df_lengths = st.session_state["strut_df"].copy()
        df_lengths.insert(0, "Day", list(range(len(df_lengths))))
        left, right = st.columns([1, 3])

        with left:
            st.markdown("### Days")
            for i in range(len(df_lengths)):
                if st.button(f"{i}", key=f"daybtn_{i}"):
                    st.session_state["selected_day"] = i
            st.download_button(
                "Download Schedule",
                data=df_lengths.to_csv(index=False).encode("utf-8"),
                file_name="tsf_strut_schedule.csv",
                mime="text/csv",
            )

        with right:
            st.dataframe(df_lengths.style.format("{:.2f}"), height=300)
            day_idx = st.session_state.get("selected_day", 0)
            st.markdown("### 3D Visualization")

            Bi_plot = ring_points(st.session_state["frame"]["prox_diam"] / 2.0, z=0.0, phase_deg=0.0)
            dist_pts_day = st.session_state["dist_pts_records"][day_idx]
            prox_center = Bi_plot.mean(axis=0)
            home_center = st.session_state["home_dist_center"]

            fig = go.Figure()

            # --- Lower ring (fixed) ---
            for s in range(6):
                a, b = Bi_plot[s], Bi_plot[(s + 1) % 6]
                cyl = cylinder_mesh(a, b, 4.5, 18)
                if cyl:
                    x, y, z, I, J, K = cyl
                    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=I.astype(int), j=J.astype(int), k=K.astype(int),
                                            color="silver", opacity=1.0, flatshading=True))

            # --- Upper ring (moving) ---
            for s in range(6):
                a, b = dist_pts_day[s], dist_pts_day[(s + 1) % 6]
                cyl = cylinder_mesh(a, b, 4.5, 18)
                if cyl:
                    x, y, z, I, J, K = cyl
                    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=I.astype(int), j=J.astype(int), k=K.astype(int),
                                            color="lightgray", opacity=1.0, flatshading=True))

            # --- Struts ---
            for s in range(6):
                a, b = Bi_plot[s], dist_pts_day[s]
                cyl = cylinder_mesh(a, b, 3.5, 14)
                if cyl:
                    x, y, z, I, J, K = cyl
                    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=I.astype(int), j=J.astype(int), k=K.astype(int),
                                            color="black", opacity=1.0, flatshading=True))

            # --- Bones: shaft + tip ---
            base_dir = (home_center - prox_center)
            base_dir /= np.linalg.norm(base_dir) + 1e-9
            # --- Dynamically compute bone length based on current ring spacing ---
            dist_center_day = dist_pts_day.mean(axis=0)
            bone_len = np.linalg.norm(dist_center_day - prox_center)

            # Natural bone colors
            prox_color = "#F5DEB3"  # wheat/light beige
            dist_color = "#8FBC8F"  # slightly darker/ivory

            # --- Proximal bone ---
            prox_start = prox_center - base_dir * (bone_len / 2)
            prox_end = prox_center + base_dir * (bone_len / 2)
            # Shaft
            shaft_end = prox_start + 0.90 * (prox_end - prox_start)
            shaft = cylinder_mesh(prox_start, shaft_end, radius=8.0, segments=28)
            if shaft:
                x, y, z, I, J, K = shaft
                fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=I.astype(int), j=J.astype(int), k=K.astype(int),
                                         color=prox_color, opacity=1.0, flatshading=True))
            # Tip
            tip = pencil_mesh(shaft_end, prox_end, base_radius=8.0, tip_radius=3.0, segments=28)
            if tip:
                x, y, z, I, J, K = tip
                fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=I.astype(int), j=J.astype(int), k=K.astype(int),
                                         color=prox_color, opacity=1.0, flatshading=True))
                

            # --- Distal bone ---
            dist_start_home = home_center - base_dir * (bone_len / 2)
            dist_end_home = home_center + base_dir * (bone_len / 2)
            rot_day = st.session_state["rot_records"][day_idx]
            home_rot = st.session_state["home_rot"]
            R_rel = rot_day.dot(home_rot.T)
            delta_center = dist_pts_day.mean(axis=0) - home_center

            def rotate_about(center, pt, R):
                return center + R.dot(pt - center)

            dist_start_day = rotate_about(home_center, dist_start_home, R_rel) + delta_center
            dist_end_day = rotate_about(home_center, dist_end_home, R_rel) + delta_center

            # Shaft
            dist_shaft_start = dist_end_day
            dist_shaft_end   = dist_start_day + 0.10 * (dist_end_day - dist_start_day)  # small upper shaft
            dist_shaft = cylinder_mesh(dist_shaft_start, dist_shaft_end, radius=8.0, segments=28)
            if dist_shaft:
                x, y, z, I, J, K = dist_shaft
                fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=I.astype(int), j=J.astype(int), k=K.astype(int),
                             color=dist_color, opacity=1.0, flatshading=True))

            # Tip (reversed)
            dist_tip_start = dist_shaft_end
            dist_tip_end   = dist_start_day
            dist_tip = pencil_mesh(dist_tip_start, dist_tip_end, base_radius=8.0, tip_radius=3.0, segments=28)
            if dist_tip:
                x, y, z, I, J, K = dist_tip
                fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=I.astype(int), j=J.astype(int), k=K.astype(int),
                             color=dist_color, opacity=1.0, flatshading=True))

            fig.update_layout(scene=dict(aspectmode='data',
                                         xaxis_title='X (mm)',yaxis_title='Y (mm)',zaxis_title='Z (mm)'),
                              height=820, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
