# Smart Gym Assistant — Evaluation Framework

This directory contains the testing and evaluation suite for the **Smart Gym Assistant**.

> **Note on Evaluation Philosophy:**
> The evaluation framework does not invent or hardcode arbitrary benchmarks. It provides scientific tooling to measure Repetition Count Error, Phase Precision/Recall/F1, Latency, and Detection Robustness on user-provided or custom-recorded footage.

---

## 1. Ground Truth Format (`ground_truth_template.csv`)

Create a CSV indexing the unseen videos to be evaluated:

| Column | Type | Description | Example Values |
|---|---|---|---|
| `video` | String | Filename of the video in `data/videos/` | `squat_01.mp4` |
| `exercise` | String | Exercise to analyze | `Squats`, `Bicep Curls` |
| `lighting` | String | Environmental lighting condition | `high`, `normal`, `low` |
| `camera_angle`| String | Camera perspective | `profile`, `frontal`, `oblique_45` |
| `occlusion` | String | Physical occlusion condition | `none`, `chair_partial`, `loose_clothing` |
| `real_reps` | Integer | Human-annotated ground truth valid repetitions | `12` |
| `phase_labels_file` | String (optional) | CSV with frame-by-frame state annotations | `labels/squat_01_phases.csv` |

### Optional Phase Labels File Format

If you wish to compute state detection Precision/Recall/F1, provide a CSV with columns:
```csv
frame,phase
0,STANDING
1,STANDING
2,DESCENDING
...
```

---

## 2. Running Evaluation via CLI

```bash
python -m evaluation.evaluate --gt evaluation/ground_truth_template.csv --videos data/videos --out evaluation/report.csv
```

Outputs:
1. `evaluation/report.csv` — Raw per-video metrics table.
2. `evaluation/report.md` — Markdown report grouping Repetition Error (MAE), FPS, and Visibility by lighting, camera angle, and occlusion.
