import sys
from pathlib import Path

# Need numpy and librosa
try:
    import numpy as np
    import librosa
    import soundfile as sf
except ImportError:
    print("Dependencies missing. Run: pip install librosa soundfile numpy")
    sys.exit(1)

def get_project_root():
    return Path(__file__).resolve().parents[2]

def add_noise(y, noise_level=0.01):
    """Adds white noise to the audio signal."""
    noise = np.random.randn(len(y))
    augmented_data = y + noise_level * noise
    return augmented_data

def shift_pitch(y, sr, n_steps=4):
    """
    Shifts pitch by n_steps (semitones). 
    Higher n_steps -> higher pitch (chipmunk).
    """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def change_speed(y, speed_factor=1.5):
    """
    Time-stretches the audio without changing pitch.
    speed_factor > 1.0 -> faster
    """
    return librosa.effects.time_stretch(y, rate=speed_factor)

def generate_augmented_audio():
    root = get_project_root()
    data_dir = root / "data"
    test_data_dir = data_dir / "test_data"
    
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all base mp4 files
    mp4_files = list(data_dir.glob("*.mp4"))
    
    if not mp4_files:
         print(f"No .mp4 files found in {data_dir}")
         return

    for mp4_file in mp4_files:
        vid_name = mp4_file.stem
        print(f"\n--- Processing {vid_name} ---")
        
        # 0. Extract base audio if needed
        base_wav = test_data_dir / f"{vid_name}_base.wav"
        if not base_wav.exists():
            import subprocess
            print(f" Extracting base audio via ffmpeg...")
            subprocess.run([
                "ffmpeg", "-i", str(mp4_file), 
                "-q:a", "0", "-map", "a", str(base_wav), "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        y, sr = librosa.load(base_wav, sr=None)
        
        # 1. High Pitch
        out_pitch = test_data_dir / f"{vid_name}_test_high_pitch.wav"
        if not out_pitch.exists():
            print(" Generating high pitch version...")
            sf.write(out_pitch, shift_pitch(y, sr, n_steps=6), sr)
        
        # 2. Fast Speed
        out_speed = test_data_dir / f"{vid_name}_test_fast.wav"
        if not out_speed.exists():
            print(" Generating fast speed version...")
            sf.write(out_speed, change_speed(y, speed_factor=1.8), sr)

        # 3. Noisy
        out_noise = test_data_dir / f"{vid_name}_test_noisy.wav"
        if not out_noise.exists():
            print(" Generating noisy version...")
            sf.write(out_noise, add_noise(y, noise_level=0.02), sr)

    print(f"\nDone! All test audio files created successfully in {test_data_dir.relative_to(root)}/")

if __name__ == "__main__":
    generate_augmented_audio()
