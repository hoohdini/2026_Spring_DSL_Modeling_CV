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
    ref_audio_dir = root / "assets" / "reference_audio"
    
    # Process all base audio files (ignoring ones we already augmented)
    base_audios = [f for f in ref_audio_dir.glob("*.wav") if "_test_" not in f.name]
    
    if not base_audios:
         print(f"No base audio files found in {ref_audio_dir}")
         return

    for target_audio in base_audios:
        print(f"\n--- Processing {target_audio.name} ---")
        y, sr = librosa.load(target_audio, sr=None)
        
        # 1. High Pitch
        out_pitch = ref_audio_dir / f"{target_audio.stem}_test_high_pitch.wav"
        if not out_pitch.exists():
            print(" Generating high pitch version...")
            sf.write(out_pitch, shift_pitch(y, sr, n_steps=6), sr)
        
        # 2. Fast Speed
        out_speed = ref_audio_dir / f"{target_audio.stem}_test_fast.wav"
        if not out_speed.exists():
            print(" Generating fast speed version...")
            sf.write(out_speed, change_speed(y, speed_factor=1.8), sr)

        # 3. Noisy
        out_noise = ref_audio_dir / f"{target_audio.stem}_test_noisy.wav"
        if not out_noise.exists():
            print(" Generating noisy version...")
            sf.write(out_noise, add_noise(y, noise_level=0.02), sr)

    print("\nDone! All test audio files created successfully in assets/reference_audio/")

if __name__ == "__main__":
    generate_augmented_audio()
