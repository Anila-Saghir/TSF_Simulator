# tsf_app.py
import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go

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

# -----------------------
# Utility math
# -----------------------
def deg2rad(d): return d * math.pi / 180.0

def rot_matrix_from_euler(roll_deg=0, pitch_deg=0, yaw_deg=0):
    r = deg2rad(roll_deg); p = deg2rad(pitch_deg); y = deg2rad(yaw_deg)
    Rx = np.array([[1,0,0],[0, math.cos(r), -math.sin(r)],[0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p),0, math.sin(p)],[0,1,0],[-math.sin(p),0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y),0],[math.sin(y), math.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx

def create_ring(radius, n=200, z=0.0):
    theta = np.linspace(0, 2*np.pi, n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(x, z)
    return np.vstack([x, y, z]).T

def transform_points(pts, R, T):
    return (R @ pts.T).T + T

def compute_strut_lengths(top_pts, bottom_pts, pairs):
    lengths = []
    for t_idx, b_idx in pairs:
        pt = top_pts[t_idx]; pb = bottom_pts[b_idx]
        lengths.append(np.linalg.norm(pt - pb))
    return np.array(lengths)

def rigid_transform(original_pts, new_pts, ring):
    c_orig = original_pts.mean(axis=0)
    c_new  = new_pts.mean(axis=0)
    P = original_pts - c_orig
    Q = new_pts - c_new
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    return (R @ (ring - c_orig).T).T + c_new

def pencil_mesh(p_base, p_tip, base_radius=12.0, tip_radius=6.0, segments=24):
    """Return mesh coordinates for a pencil-shaped bone cylinder"""
    p0, p1 = np.array(p_base, float), np.array(p_tip, float)
    axis = p1 - p0
    L = np.linalg.norm(axis)
    if L < 1e-6:
        return None, None, None, None, None, None
    zhat = axis / L
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, zhat)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    vx = np.cross(zhat, a)
    vx /= np.linalg.norm(vx)
    vy = np.cross(zhat, vx)
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
    
    base_circle = np.array([p0 + base_radius*(np.cos(t)*vx + np.sin(t)*vy) for t in angles])
    tip_circle  = np.array([p1 + tip_radius*(np.cos(t)*vx + np.sin(t)*vy) for t in angles])
    
    verts = np.vstack([base_circle, tip_circle])
    x, y, z = verts[:,0], verts[:,1], verts[:,2]
    
    I, J, K = [], [], []
    n = segments
    for i in range(n):
        i0, i1 = i, (i+1)%n
        j0, j1 = i+n, (i+1)%n+n
        I += [i0, i1]
        J += [j0, j0]
        K += [i1, j1]
    
    return x, y, z, np.array(I), np.array(J), np.array(K)

# -----------------------
# Default attachment angles & pairing
# -----------------------
default_angles_top    = np.deg2rad([10, 0, 120, 130, 240, 250])
default_angles_bottom = np.deg2rad([60, 70, 180, 190, 300, 310])
default_strut_pairs = [(0,0),(1,5),(2,1),(3,2),(4,3),(5,4)]

# -----------------------
# Initialize session_state keys if missing
# -----------------------
if "patient" not in st.session_state: st.session_state["patient"] = {}
if "deformity" not in st.session_state: st.session_state["deformity"] = {}
if "frame" not in st.session_state: st.session_state["frame"] = {}
if "plan" not in st.session_state: st.session_state["plan"] = None

# -----------------------
# Layout - Tabs
# -----------------------
tabs = st.tabs(["Patient Details", "Deformity", "Frame Dimensions", "Results"])

# -----------------------
# Tab 1: Patient
# -----------------------
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

# -----------------------
# Tab 2: Deformity inputs
# -----------------------
with tabs[1]:
    with st.form("deformity_form"):
        st.markdown('<h2 style="color:red">Define Deformities</h2>', unsafe_allow_html=True)
        with st.expander("Angular Measurements"):
            anatomy = st.selectbox("Anatomy", ["Select","Right", "Left"], index=0)
            ref_fragment = st.selectbox("Reference Fragment", ["Select", "Proximal", "Distal"], index=0)
            ap_ang = st.number_input("AP view Angulation (deg)", min_value=0.0, max_value=360.0, value=0.0, format="%.1f")
            ap_ang_type = st.radio("AP type", ["Valgus", "Varus"], horizontal=True)
            lat_ang = st.number_input("Lateral view Angulation (deg)", min_value=0.0, max_value=360.0, value=0.0, format="%.1f")
            lat_ang_type = st.radio("Lateral type", ["Apex Posterior", "Apex Anterior"], horizontal=True)
            axial_ang = st.number_input("Axial Angulation (deg)", min_value=0.0, max_value=360.0, value=0.0, format="%.1f")
            axial_ang_type = st.radio("Axial type", ["External", "Internal"], horizontal=True)
        with st.expander("Translational Measurements"):
            ap_trans = st.number_input("AP view Translation (mm)", min_value=0.0, max_value=1000.0, value=0.0, format="%.1f")
            ap_trans_type = st.radio("AP trans", ["Medial", "Lateral"], horizontal=True)
            lat_trans = st.number_input("Lateral view Translation (mm)", min_value=0.0, max_value=1000.0, value=0.0, format="%.1f")
            lat_trans_type = st.radio("Lat trans", ["Anterior", "Posterior"], horizontal=True)
            axial_trans = st.number_input("Axial Translation (mm)", min_value=0.0, max_value=1000.0, value=0.0, format="%.1f")
            axial_trans_type = st.radio("Axial trans", ["Short", "Long"], horizontal=True)
        submitted = st.form_submit_button("Enter Deformity Data")
        if submitted:
            st.session_state["deformity"] = {
                "anatomy": anatomy, "ref_fragment": ref_fragment,
                "ap_ang": float(ap_ang), "ap_ang_type": ap_ang_type,
                "lat_ang": float(lat_ang), "lat_ang_type": lat_ang_type,
                "axial_ang": float(axial_ang), "axial_ang_type": axial_ang_type,
                "ap_trans": float(ap_trans), "ap_trans_type": ap_trans_type,
                "lat_trans": float(lat_trans), "lat_trans_type": lat_trans_type,
                "axial_trans": float(axial_trans), "axial_trans_type": axial_trans_type
            }
            st.success("Deformity data saved!")

    # show mapped 6DOF
    if st.session_state.get("deformity"):
        d = st.session_state["deformity"]
        roll =  d["ap_ang"] * (1 if d["ap_ang_type"] == "Valgus" else -1)
        pitch = d["lat_ang"] * (1 if d["lat_ang_type"] == "Apex Anterior" else -1)
        yaw = d["axial_ang"] * (1 if d["axial_ang_type"] == "External" else -1)
        X = d["ap_trans"] * (1 if d["ap_trans_type"] == "Lateral" else -1)
        Y = d["lat_trans"] * (1 if d["lat_trans_type"] == "Anterior" else -1)
        Z = d["axial_trans"] * (1 if d["axial_trans_type"] == "Long" else -1)

        st.markdown("### Mapped 6 DOF (used for IK)")
        c1, c2, c3 = st.columns(3)
        c1.metric("X (mm)", f"{X:.1f}"); c1.metric("Y (mm)", f"{Y:.1f}"); c1.metric("Z (mm)", f"{Z:.1f}")
        c2.metric("Roll (deg)", f"{roll:.2f}"); c2.metric("Pitch (deg)", f"{pitch:.2f}"); c2.metric("Yaw (deg)", f"{yaw:.2f}")
        st.info("These 6 numbers represent the *target transform* of the upper fragment (top ring) relative to the reference fragment. IK will compute strut lengths that realize this transform.")

# --- Tab 3: Frame & Ring Info ---
with tabs[2]:
    st.markdown("<h2 style='color:red'>Frame & Ring Information</h2>", unsafe_allow_html=True)

    with st.expander("Ring Dimensions"):
        c1, c2 = st.columns(2)
        with c1:
            upper_ring_dia = st.number_input("Upper Ring Diameter (mm)", min_value=80.0, max_value=300.0, value=160.0)
            ring_distance = st.number_input("Distance Between Rings (mm)", min_value=40.0, max_value=400.0, value=150.0)
        with c2:
            lower_ring_dia = st.number_input("Lower Ring Diameter (mm)", min_value=80.0, max_value=300.0, value=160.0)

    with st.expander("Strut Specifications"):
        opts = ["Extra Short 75-96mm", "Short 90-125mm", "Medium 116-178mm", "Long 169-289mm"]
        # dicts to save selection, internal min/max, and day1 values
        strut_size = {}
        strut_minmax = {}
        strut_day1 = {}

        st.markdown("### Select Strut Sizes & Enter Day-1 Lengths")
        # Arrange struts in 2 rows x 3 columns
        for row in range(2):
            cols = st.columns(3)
            for col_idx, strut_num in enumerate(range(row*3+1, row*3+4)):
                with cols[col_idx]:
                    st.markdown(f"**Strut {strut_num}**")
                    # Size selectbox
                    strut_size[strut_num] = st.selectbox(f"Size", ["Select"] + opts, key=f"size_{strut_num}")
                    # Determine min,max
                    if strut_size[strut_num] == "Extra Short 75-96mm": mn, mx = 75, 96
                    elif strut_size[strut_num] == "Short 90-125mm": mn, mx = 90, 125
                    elif strut_size[strut_num] == "Medium 116-178mm": mn, mx = 116, 178
                    elif strut_size[strut_num] == "Long 169-289mm": mn, mx = 169, 289
                    else: mn, mx = 100, 200
                    strut_minmax[strut_num] = (mn, mx)
                    # Day-1 input
                    strut_day1[strut_num] = st.number_input(f"Day-1 Length", min_value=float(mn), max_value=float(mx),value=float((mn+mx)/2), key=f"day1_{strut_num}")
                    st.markdown("---")  # optional small separator
              
                
        
    if st.button("Save Frame & Strut Data"):
        st.session_state["frame"] = {
            "upper_ring_dia": upper_ring_dia,
            "lower_ring_dia": lower_ring_dia,
            "ring_distance": ring_distance,
            "strut_minmax": strut_minmax,
            "strut_day1": strut_day1,
            "strut_size": strut_size
        }
        st.success("Frame and Strut data saved!")

    # 3D preview (immediately visible if frame saved)
    if st.session_state.get("frame"):
        frame = st.session_state["frame"]
        r_top = frame["upper_ring_dia"] / 2.0
        r_bot = frame["lower_ring_dia"] / 2.0
        h = frame["ring_distance"]

        # anchor coords
        top_pts = np.array([[r_top*np.cos(a), r_top*np.sin(a), h] for a in default_angles_top])
        bot_pts = np.array([[r_bot*np.cos(a), r_bot*np.sin(a), 0.0] for a in default_angles_bottom])
        pairs = default_strut_pairs

        # Use Day-1 lengths to adjust the top ring z-coordinates (simple visualization)
        adj_top = top_pts.copy()
        # strut_day1 keys are 1..6
        day1 = frame["strut_day1"]
        for idx, (t_idx, b_idx) in enumerate(pairs, start=1):
            L = day1[idx]
            # set z of that top attachment point to L (visual heuristic)
            adj_top[t_idx, 2] = L

        # create full ring mesh and rigidly transform it to adj_top
        ring_top_mesh = create_ring(r_top, n=200, z=h)
        ring_top_trans = rigid_transform(top_pts, adj_top, ring_top_mesh)

        ring_bot_mesh = create_ring(r_bot, n=200, z=0.0)

        fig = go.Figure()
        # bottom ring
        fig.add_trace(go.Scatter3d(x=ring_bot_mesh[:,0], y=ring_bot_mesh[:,1], z=ring_bot_mesh[:,2], mode='lines', line=dict(color='red', width=8), name='Bottom Ring'))
        # top ring
        fig.add_trace(go.Scatter3d(x=ring_top_trans[:,0], y=ring_top_trans[:,1], z=ring_top_trans[:,2], mode='lines', line=dict(color='blue', width=8), name='Top Ring'))

        # struts and joints
        for idx, (t_idx, b_idx) in enumerate(pairs, start=1):
            pt = adj_top[t_idx]
            pb = bot_pts[b_idx]
            fig.add_trace(go.Scatter3d(x=[pt[0], pb[0]], y=[pt[1], pb[1]], z=[pt[2], pb[2]], mode='lines', line=dict(color='green', width=10), name=f"Strut {idx}"))
            fig.add_trace(go.Scatter3d(x=[pt[0]], y=[pt[1]], z=[pt[2]], mode='markers', marker=dict(size=5, color='black'), showlegend=False))
            fig.add_trace(go.Scatter3d(x=[pb[0]], y=[pb[1]], z=[pb[2]], mode='markers', marker=dict(size=5, color='black'), showlegend=False))

        fig.update_layout(scene=dict(aspectmode='cube'), height=600, title="TSF Fixator - Day 1 Preview")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Tab 4: Results (Final Updated Version — Fixed vs Moving Bone)
# -----------------------
with tabs[3]:
    st.markdown('<h2 style="color:darkred">Results</h2>', unsafe_allow_html=True)

    if not st.session_state.get("frame"):
        st.warning("Please set Frame & Ring parameters in Tab 3 first.")
    elif not st.session_state.get("deformity"):
        st.warning("Please enter Deformity data in Tab 2 first.")
    else:
        frame = st.session_state["frame"]
        d = st.session_state["deformity"]

        # --- Map deformity to 6-DOF target transform ---
        roll  = d["ap_ang"]  * (1 if d["ap_ang_type"]  == "Valgus" else -1)
        pitch = d["lat_ang"] * (1 if d["lat_ang_type"] == "Apex Anterior" else -1)
        yaw   = d["axial_ang"] * (1 if d["axial_ang_type"] == "External" else -1)
        X = d["ap_trans"]  * (1 if d["ap_trans_type"]  == "Lateral" else -1)
        Y = d["lat_trans"] * (1 if d["lat_trans_type"] == "Anterior" else -1)
        Z = d["axial_trans"] * (1 if d["axial_trans_type"] == "Long" else -1)

        st.write("Target transform (upper fragment relative to reference):")
        st.write(f"Translation (mm): X={X:.1f}, Y={Y:.1f}, Z={Z:.1f}")
        st.write(f"Rotation (deg): roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")

        # --- Base geometry ---
        r_top = frame["upper_ring_dia"] / 2.0
        r_bot = frame["lower_ring_dia"] / 2.0
        sep   = frame["ring_distance"]

        top_attach    = np.array([[r_top*np.cos(a), r_top*np.sin(a), sep] for a in default_angles_top])
        bottom_attach = np.array([[r_bot*np.cos(a), r_bot*np.sin(a), 0.0] for a in default_angles_bottom])

        # --- Strut lengths ---
        init_lengths = np.array([frame["strut_day1"][i] for i in range(1,7)], dtype=float)
        minL = np.array([frame["strut_minmax"][i][0] for i in range(1,7)], dtype=float)
        maxL = np.array([frame["strut_minmax"][i][1] for i in range(1,7)], dtype=float)

        # --- Target transform (corrected alignment) ---
        R_target = rot_matrix_from_euler(roll, pitch, yaw)
        T_target = np.array([X, Y, Z])
        top_attach_target = transform_points(top_attach, R_target, T_target)
        target_lengths = compute_strut_lengths(top_attach_target, bottom_attach, default_strut_pairs)

        # --- Plan generation ---
        st.markdown("### Generate plan")
        c1, c2 = st.columns([1, 1])
        with c1:
            num_days = st.number_input("Number of days for correction", min_value=1, max_value=365, value=30, step=1)
        with c2:
            generate = st.button("Generate daily Plan")

        if generate:
            days = int(num_days)
            plan = []

            # --- Step 1 : starting (deformed) configuration ---
            R_deformed = rot_matrix_from_euler(-roll, -pitch, -yaw)
            T_deformed = np.array([-X, -Y, -Z])
            top_attach_deformed = transform_points(top_attach, R_deformed, T_deformed)
            start_lengths = compute_strut_lengths(top_attach_deformed, bottom_attach, default_strut_pairs)

            # --- Step 2 : interpolate from deformed → corrected ---
            for day in range(1, days+1):
                t = (day-1)/(days-1) if days>1 else 1.0
                lengths_day = start_lengths*(1-t) + target_lengths*t
                lengths_day = np.minimum(np.maximum(lengths_day, minL), maxL)
                plan.append(lengths_day.copy())

            st.session_state["plan"] = {
                "days": days,
                "plan_lengths": np.array(plan),
                "top_attach": top_attach,
                "bottom_attach": bottom_attach,
                "pairs": default_strut_pairs,
                "R_target": R_target,
                "T_target": T_target,
                "R_deformed": R_deformed,
                "T_deformed": T_deformed
            }
            st.success("IK plan generated: starts from deformed → corrected.")

        plan_obj = st.session_state.get("plan")
        if plan_obj:
            days = plan_obj["days"]
            st.markdown(f"### Plan Summary — {days} days")
            df = pd.DataFrame(plan_obj["plan_lengths"], columns=[f"Strut {i+1}" for i in range(6)])
            df.index = [f"Day {i+1}" for i in range(df.shape[0])]
            st.dataframe(df.style.format("{:.1f}"))

            st.markdown("Select a day to preview TSF and bones:")
            selected_day = st.number_input("Pick day", min_value=1, max_value=days, value=1, step=1)
            if st.button("Show selected day") or st.session_state.get("_tmp_selected"):
                selected = int(selected_day) - 1
                if st.session_state.get("_tmp_selected"):
                    selected = st.session_state.pop("_tmp_selected") - 1

                lengths_day = plan_obj["plan_lengths"][selected]
                top_base = plan_obj["top_attach"].copy()
                bottom   = plan_obj["bottom_attach"].copy()
                R_start  = plan_obj["R_deformed"]
                T_start  = plan_obj["T_deformed"]
                R_end    = plan_obj["R_target"]
                T_end    = plan_obj["T_target"]

                # --- interpolate transform between deformed & corrected ---
                t = selected / (plan_obj["days"] - 1) if plan_obj["days"] > 1 else 1.0
                T_interp = (1 - t) * T_start + t * T_end
                R_interp = R_start @ rot_matrix_from_euler(t*roll, t*pitch, t*yaw)
                top_trans = transform_points(top_base, R_interp, T_interp)

                # adjust top points to match current strut lengths
                top_pts_day = top_trans.copy()
                for i, (t_idx, b_idx) in enumerate(plan_obj["pairs"]):
                    pb = bottom[b_idx]
                    vec = top_trans[t_idx] - pb
                    cur_len = np.linalg.norm(vec)
                    if cur_len < 1e-6:
                        vec = np.array([0,0,1.0]); cur_len = 1.0
                    top_pts_day[t_idx] = pb + (vec/cur_len) * lengths_day[i]

                # --- 3D Plot ---
                fig = go.Figure()

                # Rings (rigid transform so top ring attaches correctly)
                bottom_ring = create_ring(r_bot, n=200, z=0.0)
                top_ring    = create_ring(r_top, n=200, z=sep)
                top_ring_moved = rigid_transform(top_base, top_pts_day, top_ring)

                fig.add_trace(go.Scatter3d(
                    x=bottom_ring[:,0], y=bottom_ring[:,1], z=bottom_ring[:,2],
                    mode='lines', line=dict(width=6, color='red'), name='Bottom Ring'
                ))
                fig.add_trace(go.Scatter3d(
                    x=top_ring_moved[:,0], y=top_ring_moved[:,1], z=top_ring_moved[:,2],
                    mode='lines', line=dict(width=6, color='blue'), name='Top Ring'
                ))

                # Struts
                for i, (t_idx, b_idx) in enumerate(plan_obj["pairs"]):
                    pt = top_pts_day[t_idx]; pb = bottom[b_idx]
                    fig.add_trace(go.Scatter3d(
                        x=[pt[0], pb[0]], y=[pt[1], pb[1]], z=[pt[2], pb[2]],
                        mode='lines', line=dict(width=12, color='green'), name=f"Strut {i+1}"
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=[(pt[0]+pb[0])/2.0], y=[(pt[1]+pb[1])/2.0], z=[(pt[2]+pb[2])/2.0],
                        mode='text', text=[f"{lengths_day[i]:.1f} mm"], showlegend=False
                    ))

                                
                # --- Bone visualization (bottom fixed, top moves, auto-scaled) ---
                tip_clearance_mm = 0.5
                bone_base_radius = 12.0
                bone_tip_radius  = 4.0
                bone_segments    = 24

                # auto-scale bone length = 1/3 of current ring distance
                bone_length = frame["ring_distance"] / 3.0

                def add_bone(fig, p_base, p_tip, base_r, tip_r, color):
                    x, y, z, I, J, K = pencil_mesh(p_base, p_tip, base_r, tip_r, bone_segments)
                    if x is not None:
                        fig.add_trace(go.Mesh3d(
                            x=x, y=y, z=z, i=I, j=J, k=K,
                            color=color, opacity=0.85
                        ))

                # --- Bottom (fixed) bone ---
                distal_centroid = bottom.mean(axis=0)
                distal_axis = np.array([0, 0, 1.0])  # up along Z
                distal_tip = distal_centroid + distal_axis * bone_length
                add_bone(fig, distal_centroid, distal_tip,
                         bone_base_radius, bone_tip_radius, 'lightblue')

                # --- Top (moving) bone ---
                proximal_centroid = top_pts_day.mean(axis=0)
                proximal_axis = np.array([0, 0, -1.0])  # down along Z
                proximal_tip = proximal_centroid + proximal_axis * bone_length
                add_bone(fig, proximal_centroid, proximal_tip,
                         bone_base_radius, bone_tip_radius, 'lightgreen')

                                # --- Layout ---
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(title='X'),
                        yaxis=dict(title='Y'),
                        zaxis=dict(title='Z'),
                        aspectmode='auto'
                    ),
                    width=1000, height=700,
                    title=f"Day {selected+1} TSF & Bones"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Quick-day buttons
            max_buttons = min(12, plan_obj["days"])
            bcols = st.columns(max_buttons)
            for i in range(max_buttons):
                if bcols[i].button(f"D{i+1}"):
                    st.session_state._tmp_selected = i+1
                    st.experimental_rerun()
        else:
            st.info("Generate the plan to see daily interpolation and 3D preview.")

