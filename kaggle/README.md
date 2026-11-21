# Running Smart Gym Assistant on Kaggle

This directory contains resources to execute the **Smart Gym Assistant** inside Kaggle Notebooks and GPU/CPU environments.

---

## 1. Quick Start on Kaggle (3 Steps)

### Step 1: Upload or Clone the Project to Kaggle
In a new Kaggle Notebook:
```python
!git clone https://github.com/<your-username>/smart-gym-assistant.git
%cd smart-gym-assistant
```
*Or upload the project zip as a Kaggle Dataset and attach it to your notebook.*

### Step 2: Install Dependencies
```python
!pip install -q mediapipe==0.10.14 "numpy<2.0.0" opencv-python-headless plotly pyyaml scipy
```
> **Tip for Kaggle:** We use `opencv-python-headless` instead of `opencv-python` to prevent any missing X11/libGL GUI server errors on headless Linux.

### Step 3: Run the Analysis Pipeline
```python
from run_kaggle import process_video_headless

results = process_video_headless(
    video_path="data/videos/squat_sample.mp4",
    exercise_name="Squats",
    output_video_path="/kaggle/working/annotated_squat.mp4",
    output_json_path="/kaggle/working/session.json",
    output_csv_path="/kaggle/working/reps.csv"
)
```

---

## 2. Using the Ready-to-Run Kaggle Notebook (`smart_gym_kaggle.ipynb`)

1. Go to [kaggle.com/code](https://www.kaggle.com/code).
2. Click **New Notebook** -> **File** -> **Import Notebook**.
3. Upload `kaggle/smart_gym_kaggle.ipynb` (or `kaggle_notebook.ipynb` from repository root).
4. Click **Run All**.
5. The notebook will:
   - Run the unit test suite (`pytest -q`)
   - Process the sample workout video with MediaPipe Pose & the Biomechanical FSM
   - Embed and play the annotated video with the live skeletal HUD directly inside the notebook
   - Render interactive Plotly angle curves & quality radar charts
   - Output `session.json` and `reps.csv` to `/kaggle/working/` for download

---

## 3. Running Interactive Streamlit from Kaggle (Cloudflare Tunnel)

To access the full multi-page Streamlit web application from your browser while running on Kaggle:

In a notebook cell:
```python
import subprocess, time, os

# 1. Start Streamlit in the background
st_proc = subprocess.Popen(["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.headless=true"])
time.sleep(3)

# 2. Download and launch Cloudflare Tunnel
if not os.path.exists("cloudflared"):
    !wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    !chmod +x cloudflared

print("Opening secure public tunnel...")
!./cloudflared tunnel --url http://localhost:8501
```
Cloudflare will output a public URL (e.g. `https://xxxx.trycloudflare.com`). Click the link to open the Streamlit UI in your browser!
