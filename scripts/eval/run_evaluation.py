import os
import sys
import time
import csv
from pathlib import Path
import cv2

# Ensure we can import our pipeline modules
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Video dependencies
import mediapipe as mp
from src.multimodal_coach.pipelines.vision.pose_analyzer import PoseAnalyzer, LEARNED_PUNCH_PATH
from src.multimodal_coach.pipelines.vision.expression import ExpressionAnalyzer

# Audio dependencies
try:
    from src.multimodal_coach.pipelines.audio.audio_analyzer import AudioAnalyzer
except ImportError:
    AudioAnalyzer = None

# Scoring helpers (adapted from runner.py)
def compute_pose_score(metrics) -> float:
    body  = max(0.0, 1.0 - metrics.body_tilt_angle / 45.0)
    head  = max(0.0, 1.0 - metrics.head_tilt_angle / 50.0)
    tremor = 1.0 - metrics.tremor_level
    return (body + head + tremor) / 3.0

def compute_audio_score(wpm: float, energy: float, pitch_std: float) -> float:
    if wpm <= 0 and energy <= 0 and pitch_std <= 0: return 0.5
    wpm_score = max(0.0, 1.0 - ((wpm - 125.0) / 50.0) ** 2) if wpm > 0 else 0.5
    if pitch_std <= 0: pitch_score = 0.5
    elif pitch_std < 30: pitch_score = pitch_std / 30.0
    elif pitch_std <= 65: pitch_score = 1.0
    else: pitch_score = max(0.0, 1.0 - (pitch_std - 65.0) / 65.0)
    
    if energy <= 0: energy_score = 0.5
    elif energy < 0.02: energy_score = energy / 0.02
    elif energy <= 0.05: energy_score = 1.0
    else: energy_score = max(0.0, 1.0 - (energy - 0.05) / 0.05)
    return (wpm_score + pitch_score + energy_score) / 3.0

# Vessl / Headless Evaluator
class HeadlessEvaluator:
    def __init__(self):
        self.pose_analyzer = PoseAnalyzer(learned_punch_path=LEARNED_PUNCH_PATH)
        self.expression_analyzer = ExpressionAnalyzer()
        self.mp_holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    def evaluate_video(self, video_path: str):
        """Processes a video frame-by-frame as fast as possible (no cv2.imshow)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None
            
        pose_scores, expr_scores = [], []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_holistic.process(frame_rgb)
            
            # 1. Pose
            metrics = self.pose_analyzer.analyze(
                results.pose_landmarks,
                results.left_hand_landmarks,
                results.right_hand_landmarks,
            )
            pose_scores.append(compute_pose_score(metrics))
            
            # 2. Expression
            expr_metrics = self.expression_analyzer.analyze(results.face_landmarks)
            if expr_metrics:
                expr_scores.append(expr_metrics.confidence_score)
                
        cap.release()
        
        avg_pose = sum(pose_scores) / len(pose_scores) if pose_scores else 0
        avg_expr = sum(expr_scores) / len(expr_scores) if expr_scores else 0
        return avg_pose * 100, avg_expr * 100

    def evaluate_audio(self, audio_path: str):
        try:
            import librosa
            import numpy as np
            import whisper
            
            y, sr = librosa.load(audio_path, sr=16000)
            
            # 1. WPM via Whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, fp16=False, language="ko")
            words = len(result.get("text", "").strip().split())
            dur = len(y) / sr
            wpm = words / (dur / 60.0) if dur > 0 else 0.0
            
            # 2. Energy
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            energy = float(np.mean(rms))
            
            # 3. Pitch
            y_pitch = librosa.resample(y, orig_sr=sr, target_sr=8000) if sr > 8000 else y
            f0 = librosa.yin(y_pitch, fmin=70, fmax=400, sr=8000, hop_length=1024)
            # Yin doesn't return voiced probabilities natively, so we mask via RMS energy
            rms_pitch = librosa.feature.rms(y=y_pitch, frame_length=2048, hop_length=1024)[0]
            f0_voiced = f0[:len(rms_pitch)][rms_pitch > 0.01]
            pitch = float(np.nanstd(f0_voiced)) if len(f0_voiced) > 0 else 0.0
            
            score = compute_audio_score(wpm, energy, pitch) * 100
            return score, wpm, energy, pitch
        except Exception as e:
            print(f"Error evaluating audio {audio_path}: {e}")
            return 0, 0, 0, 0

def run_all_tests():
    evaluator = HeadlessEvaluator()
    results = []
    
    audio_dir = REPO_ROOT / "assets" / "reference_audio"
    video_dir = REPO_ROOT / "data"
    
    # 1. Automatically find all test audio files
    audio_test_files = sorted(list(audio_dir.glob("*_test_*.wav")))
    # Add base files just for baseline comparison
    base_audios = [f for f in audio_dir.glob("*.wav") if "_test_" not in f.name]
    audio_test_files = base_audios + audio_test_files
    
    print(f"--- Starting Headless Audio Evaluation ({len(audio_test_files)} files) ---")
    for p in audio_test_files:
        score, wpm, energy, pitch = evaluator.evaluate_audio(str(p))
        results.append({
            "type": "AUDIO",
            "filename": p.name,
            "overall_score": f"{score:.1f}",
            "metric_1 (WPM/Pose)": f"{wpm:.1f}",
            "metric_2 (Energy/Expr)": f"{energy:.4f}",
            "metric_3 (Pitch)": f"{pitch:.1f}"
        })
        print(f"Evaluated {p.name}: Score={score:.1f} | WPM={wpm:.1f}")
            
    # 2. Automatically find all videos in data/
    video_test_files = sorted(list(video_dir.glob("*.mp4")))
    
    print(f"\n--- Starting Headless Video Evaluation ({len(video_test_files)} files) ---")
    for p in video_test_files:
        s_pose, s_expr = evaluator.evaluate_video(str(p))
        results.append({
            "type": "VIDEO",
            "filename": p.name,
            "overall_score": "N/A",
            "metric_1 (WPM/Pose)": f"{s_pose:.1f}",
            "metric_2 (Energy/Expr)": f"{s_expr:.1f}",
            "metric_3 (Pitch)": "N/A"
        })
        print(f"Evaluated {p.name}: Pose={s_pose:.1f} | Expr={s_expr:.1f}")

    # Output to CSV
    csv_path = REPO_ROOT / "scripts" / "eval" / "results.csv"
    keys = results[0].keys() if results else []
    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nAll evaluations complete! Results saved to: {csv_path.relative_to(REPO_ROOT)}")

if __name__ == "__main__":
    run_all_tests()
