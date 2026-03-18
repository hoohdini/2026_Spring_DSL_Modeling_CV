import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Ensure we can import our pipeline modules
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.multimodal_coach.pipelines.vision.expression import ExpressionAnalyzer
from src.multimodal_coach.pipelines.vision.pose_analyzer import LEARNED_PUNCH_PATH, PoseAnalyzer


def compute_pose_score(metrics) -> float:
    body = max(0.0, 1.0 - metrics.body_tilt_angle / 45.0)
    head = max(0.0, 1.0 - metrics.head_tilt_angle / 50.0)
    tremor = 1.0 - metrics.tremor_level
    return (body + head + tremor) / 3.0


def compute_audio_score(wpm: float, energy: float, pitch_std: float) -> float:
    if wpm <= 0 and energy <= 0 and pitch_std <= 0:
        return 0.5

    wpm_score = max(0.0, 1.0 - ((wpm - 125.0) / 50.0) ** 2) if wpm > 0 else 0.5

    if pitch_std <= 0:
        pitch_score = 0.5
    elif pitch_std < 30:
        pitch_score = pitch_std / 30.0
    elif pitch_std <= 65:
        pitch_score = 1.0
    else:
        pitch_score = max(0.0, 1.0 - (pitch_std - 65.0) / 65.0)

    if energy <= 0:
        energy_score = 0.5
    elif energy < 0.02:
        energy_score = energy / 0.02
    elif energy <= 0.05:
        energy_score = 1.0
    else:
        energy_score = max(0.0, 1.0 - (energy - 0.05) / 0.05)

    return (wpm_score + pitch_score + energy_score) / 3.0


class HeadlessEvaluator:
    def __init__(
        self,
        whisper_model_name: str = "base",
        language: str = "ko",
        use_whisper: bool = True,
        frame_stride: int = 3,
    ):
        self.pose_analyzer = PoseAnalyzer(learned_punch_path=LEARNED_PUNCH_PATH)
        self.expression_analyzer = ExpressionAnalyzer()
        self.mp_holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.language = language
        self.use_whisper = use_whisper
        self.frame_stride = max(1, int(frame_stride))
        self._audio_enabled = False
        self._whisper_model = None
        self._init_audio_modules(whisper_model_name)

    def _init_audio_modules(self, whisper_model_name: str) -> None:
        try:
            import librosa
            import numpy as np
        except ImportError as exc:
            print(f"[WARN] Audio dependencies missing ({exc}). Audio evaluation will be skipped.")
            return

        self._librosa = librosa
        self._np = np
        if not self.use_whisper:
            self._audio_enabled = True
            print("[INFO] Whisper disabled. WPM will use neutral fallback (125).")
            return

        try:
            import whisper
            self._whisper_model = whisper.load_model(whisper_model_name)
            self._audio_enabled = True
            print(f"[INFO] Whisper model loaded: {whisper_model_name}")
        except Exception as exc:
            print(f"[WARN] Failed to load Whisper model '{whisper_model_name}': {exc}")

    def evaluate_video(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None

        pose_scores, expr_scores = [], []
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            if frame_index % self.frame_stride != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_holistic.process(frame_rgb)

            metrics = self.pose_analyzer.analyze(
                results.pose_landmarks,
                results.left_hand_landmarks,
                results.right_hand_landmarks,
            )
            pose_scores.append(compute_pose_score(metrics))

            expr_metrics = self.expression_analyzer.analyze(results.face_landmarks)
            if expr_metrics:
                expr_scores.append(expr_metrics.confidence_score)

        cap.release()

        avg_pose = sum(pose_scores) / len(pose_scores) if pose_scores else 0.0
        avg_expr = sum(expr_scores) / len(expr_scores) if expr_scores else 0.0
        return avg_pose * 100.0, avg_expr * 100.0

    def evaluate_audio(self, audio_path: Path):
        if not self._audio_enabled:
            return None, None, None, None

        try:
            y, sr = self._librosa.load(str(audio_path), sr=16000)

            if self.use_whisper and self._whisper_model is not None:
                result = self._whisper_model.transcribe(str(audio_path), fp16=False, language=self.language)
                words = len(result.get("text", "").strip().split())
                dur = len(y) / sr
                wpm = words / (dur / 60.0) if dur > 0 else 0.0
            else:
                # Neutral fallback so score is driven by acoustic features.
                wpm = 125.0

            rms = self._librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            energy = float(self._np.mean(rms))

            y_pitch = self._librosa.resample(y, orig_sr=sr, target_sr=8000) if sr > 8000 else y
            f0 = self._librosa.yin(y_pitch, fmin=70, fmax=400, sr=8000, hop_length=1024)
            rms_pitch = self._librosa.feature.rms(y=y_pitch, frame_length=2048, hop_length=1024)[0]
            f0_voiced = f0[: len(rms_pitch)][rms_pitch > 0.01]
            pitch = float(self._np.nanstd(f0_voiced)) if len(f0_voiced) > 0 else 0.0

            score = compute_audio_score(wpm, energy, pitch) * 100.0
            return score, wpm, energy, pitch
        except Exception as exc:
            print(f"[WARN] Error evaluating audio {audio_path.name}: {exc}")
            return None, None, None, None


def _safe_mean(values):
    return statistics.mean(values) if values else None


def _safe_stdev(values):
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _clip_0_100(value: float) -> float:
    return max(0.0, min(100.0, value))


def _parse_audio_variant(stem: str):
    # Expected stem examples:
    # - clinton1_base
    # - clinton1_test_fast
    # - clinton1_test_high_pitch
    # - clinton1_test_noisy
    m = re.match(r"(.+)_(base|test_fast|test_high_pitch|test_noisy)$", stem)
    if not m:
        return stem, "unknown"
    return m.group(1), m.group(2)


def _build_audio_reliability(audio_rows):
    by_clip = {}
    for row in audio_rows:
        clip_id, variant = _parse_audio_variant(Path(row["filename"]).stem)
        row["clip_id"] = clip_id
        row["variant"] = variant
        by_clip.setdefault(clip_id, {})[variant] = row

    per_clip = []
    for clip_id, variants in sorted(by_clip.items()):
        base = variants.get("base")
        if base is None:
            continue

        base_score = float(base["overall_score"])
        compared_scores = [base_score]
        deltas = []
        for v_name in ("test_fast", "test_high_pitch", "test_noisy"):
            row = variants.get(v_name)
            if not row:
                continue
            score = float(row["overall_score"])
            compared_scores.append(score)
            deltas.append(abs(score - base_score))

        mean_delta = _safe_mean(deltas)
        std_dev = _safe_stdev(compared_scores)

        # Heuristic reliability:
        # - robustness: less change from base under perturbation is better
        # - stability: lower score spread across variants is better
        robustness = _clip_0_100(100.0 - (mean_delta if mean_delta is not None else 100.0))
        stability = _clip_0_100(100.0 - std_dev * 2.0)
        reliability = _clip_0_100(0.7 * robustness + 0.3 * stability)

        per_clip.append(
            {
                "type": "AUDIO",
                "clip_id": clip_id,
                "base_score": f"{base_score:.2f}",
                "avg_abs_delta_from_base": f"{(mean_delta or 0.0):.2f}",
                "score_stddev": f"{std_dev:.2f}",
                "robustness_subscore": f"{robustness:.2f}",
                "stability_subscore": f"{stability:.2f}",
                "reliability_score": f"{reliability:.2f}",
            }
        )

    overall = _safe_mean([float(r["reliability_score"]) for r in per_clip])
    return per_clip, overall


def _build_video_reliability(video_rows):
    if not video_rows:
        return [], None

    composites = []
    per_video = []
    for row in video_rows:
        pose = float(row["metric_1 (WPM/Pose)"])
        expr = float(row["metric_2 (Energy/Expr)"])
        composite = (pose + expr) / 2.0
        composites.append(composite)
        per_video.append(
            {
                "type": "VIDEO",
                "clip_id": Path(row["filename"]).stem,
                "base_score": "N/A",
                "avg_abs_delta_from_base": "N/A",
                "score_stddev": "N/A",
                "robustness_subscore": "N/A",
                "stability_subscore": "N/A",
                "reliability_score": f"{composite:.2f}",
            }
        )

    std_dev = _safe_stdev(composites)
    overall_stability = _clip_0_100(100.0 - std_dev * 2.0)
    return per_video, overall_stability


def run_all_tests(args):
    print("[INFO] Initializing evaluator...")
    evaluator = HeadlessEvaluator(
        whisper_model_name=args.whisper_model,
        language=args.language,
        use_whisper=not args.disable_whisper,
        frame_stride=args.frame_stride,
    )
    results = []

    audio_dir = Path(args.audio_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_test_files = sorted(audio_dir.glob("*.wav"))
    print(f"--- Starting Headless Audio Evaluation ({len(audio_test_files)} files) ---")
    for p in audio_test_files:
        score, wpm, energy, pitch = evaluator.evaluate_audio(p)
        if score is None:
            continue
        results.append(
            {
                "type": "AUDIO",
                "filename": p.name,
                "overall_score": f"{score:.1f}",
                "metric_1 (WPM/Pose)": f"{wpm:.1f}",
                "metric_2 (Energy/Expr)": f"{energy:.4f}",
                "metric_3 (Pitch)": f"{pitch:.1f}",
            }
        )
        print(f"Evaluated {p.name}: Score={score:.1f} | WPM={wpm:.1f}")

    video_test_files = sorted(video_dir.glob("*.mp4"))
    print(f"\n--- Starting Headless Video Evaluation ({len(video_test_files)} files) ---")
    for p in video_test_files:
        s_pose, s_expr = evaluator.evaluate_video(p)
        if s_pose is None or s_expr is None:
            print(f"[WARN] Failed to open video: {p.name}")
            continue
        results.append(
            {
                "type": "VIDEO",
                "filename": p.name,
                "overall_score": "N/A",
                "metric_1 (WPM/Pose)": f"{s_pose:.1f}",
                "metric_2 (Energy/Expr)": f"{s_expr:.1f}",
                "metric_3 (Pitch)": "N/A",
            }
        )
        print(f"Evaluated {p.name}: Pose={s_pose:.1f} | Expr={s_expr:.1f}")

    detailed_csv_path = output_dir / "results.csv"
    if results:
        keys = list(results[0].keys())
        with detailed_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    else:
        with detailed_csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "filename", "overall_score", "metric_1 (WPM/Pose)", "metric_2 (Energy/Expr)", "metric_3 (Pitch)"])

    audio_rows = [r for r in results if r["type"] == "AUDIO"]
    video_rows = [r for r in results if r["type"] == "VIDEO"]
    audio_reliability_rows, audio_reliability_overall = _build_audio_reliability(audio_rows)
    video_reliability_rows, video_reliability_overall = _build_video_reliability(video_rows)
    reliability_rows = audio_reliability_rows + video_reliability_rows

    reliability_csv_path = output_dir / "reliability_summary.csv"
    summary_headers = [
        "type",
        "clip_id",
        "base_score",
        "avg_abs_delta_from_base",
        "score_stddev",
        "robustness_subscore",
        "stability_subscore",
        "reliability_score",
    ]
    with reliability_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers)
        writer.writeheader()
        writer.writerows(reliability_rows)

    summary_payload = {
        "audio_reliability_mean": round(audio_reliability_overall, 2) if audio_reliability_overall is not None else None,
        "video_stability_score": round(video_reliability_overall, 2) if video_reliability_overall is not None else None,
        "num_audio_files_evaluated": len(audio_rows),
        "num_video_files_evaluated": len(video_rows),
        "num_audio_clips_grouped": len(audio_reliability_rows),
        "notes": "Audio reliability is computed from base-vs-augmented robustness and score stability; video score reports per-file composite and global dispersion-based stability.",
    }
    reliability_json_path = output_dir / "reliability_summary.json"
    reliability_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"\nDetailed results: {detailed_csv_path.relative_to(REPO_ROOT)}")
    print(f"Reliability CSV: {reliability_csv_path.relative_to(REPO_ROOT)}")
    print(f"Reliability JSON: {reliability_json_path.relative_to(REPO_ROOT)}")
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Headless evaluation + reliability report generator.")
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "test_data"),
        help="Directory containing test audio (.wav).",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "test_data"),
        help="Directory containing test videos (.mp4).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "scripts" / "eval"),
        help="Directory to save csv/json outputs.",
    )
    parser.add_argument("--language", type=str, default="ko", help="Whisper transcription language.")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model name.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="Process every Nth frame for faster video evaluation (>=1).",
    )
    parser.add_argument(
        "--disable-whisper",
        action="store_true",
        help="Skip Whisper transcription and use neutral WPM fallback for faster runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_all_tests(parse_args())
