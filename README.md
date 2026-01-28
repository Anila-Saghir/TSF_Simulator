**Taylor Spatial Frame Simulator**

This is an interactive web-based application for simulating and planning Taylor Spatial Frame corrections. The simulator enables entry of patient deformity parameters, frame geometry, and strut lengths, then generates daily adjustment plans and visual 3D corrections step by step.

## 🎯 Features

- **Patient and Frame Data Input**
  - Enter patient details, deformity angles/translations, and ring dimensions.
  - Define six struts with individual size ranges and Day-1 lengths.

- **3D Visualization**
  - Fully interactive 3D model using Plotly.
  - View bottom and top rings, struts, and realistic bone geometry.
  - Distinct strut colors and visible joint rings at attachment points.

- **Daily Correction Plan**
  - Automatically generates strut length plans over user-defined days.
  - Interpolates deformity correction gradually toward the target.
  - Displays tabular daily plan and dynamic 3D simulation.


**Accessing the App**

You can access the live app here: [(https://tsf-simulator-madeby-anila-saghir-2026.streamlit.app/)]
No installation is required — it runs directly in your web browser.

**How to Use the App**
The app is organized into four tabs:

**Patient Data**
Enter patient information such as Name, Age, Operation Code, and Operation Date.
Click Enter Patient Data to save.

**Define Deformities**
Input angular and translational deformities for AP, lateral, and axial views.
Click Enter Deformity Data to save.

**Select Frame**
Choose the dimensions of the frame rings and strut types. Enter frame offsets and operative mode.
Click Enter Frame Selection to save.

**Results**
Enter the number of correction days.
Click Run Simulation to compute strut schedules and 3D visualization.
You can select each day to view bone alignment and download the strut schedule as a CSV file.

**Tips**
Make sure all previous tabs are completed before running the simulation.
The 3D visualization is interactive — you can rotate, zoom, and pan.
Download the strut schedule for offline reference.

**Support**

For any questions or assistance, contact:
Anila Saghir
Email: asaghir@ssuet.edu.pk
