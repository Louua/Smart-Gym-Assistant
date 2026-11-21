# Smart Gym Assistant — Evaluation Report

## Overall Benchmark Summary

- **Total Videos Evaluated:** 3
- **Mean Absolute Repetition Error (MAE):** 2.00 reps
- **Average Processing Speed:** 24.3 FPS
- **Average Landmark Visibility:** 0.321

## Repetition Error by Testing Condition

### Grouped by `lighting`

| lighting | count | mae | avg_fps | avg_vis |
| --- | --- | --- | --- | --- |
| high | 3 | 2.0 | 24.3 | 0.321 |

### Grouped by `camera_angle`

| camera_angle | count | mae | avg_fps | avg_vis |
| --- | --- | --- | --- | --- |
| profile | 3 | 2.0 | 24.3 | 0.321 |

### Grouped by `occlusion`

| occlusion | count | mae | avg_fps | avg_vis |
| --- | --- | --- | --- | --- |
| none | 3 | 2.0 | 24.3 | 0.321 |

## Detailed Video Results

| video | exercise | lighting | camera_angle | real_reps | predicted_reps | rep_error | fps | overall_form |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| squat_sample.mp4 | Squats | high | profile | 2 | 2 | 0 | 24.5 | 92.1 |
| demo_squat.mp4 | Squats | high | profile | 3 | 0 | 3 | 24.5 | 30.0 |
| demo_curl.mp4 | Bicep Curls | high | profile | 3 | 0 | 3 | 23.9 | 30.0 |
