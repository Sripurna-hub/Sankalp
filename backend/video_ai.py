import cv2
import os
import tempfile

from backend.image_ai import run_accurate_ai_inference

def analyze_video(video_path, sample_every_n_frames=30):
    """
    Video processing module.

    Extracts sampled frames from a video and runs the same existing
    image-analysis engine on each sampled frame. The overall video
    severity is the maximum sampled-frame severity.

    This module is kept separate so video behavior can be improved
    independently later.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    frame_index = 0
    results = []

    temp_dir = tempfile.mkdtemp(prefix="disaster_video_frames_")

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % sample_every_n_frames == 0:
            frame_path = os.path.join(temp_dir, f"frame_{frame_index:06d}.jpg")
            cv2.imwrite(frame_path, frame)

            category, severity, annotated_path = run_accurate_ai_inference(frame_path)
            results.append({
                "frame": frame_index,
                "time_seconds": frame_index / fps,
                "category": category,
                "severity": severity,
                "annotated_path": annotated_path
            })

        frame_index += 1

    cap.release()

    if not results:
        return {
            "category": "No analyzable frames",
            "severity": 0,
            "annotated_path": None,
            "frames_analyzed": 0,
            "frame_results": []
        }

    worst = max(results, key=lambda x: x["severity"])

    return {
        "category": worst["category"],
        "severity": worst["severity"],
        "annotated_path": worst["annotated_path"],
        "frames_analyzed": len(results),
        "frame_results": results
    }
