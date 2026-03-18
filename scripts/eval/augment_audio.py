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
    target_audio = ref_audio_dir / "obama1.wav"
    
    if not target_audio.exists():
         print(f"Target audio not found: {target_audio}")
         print("Make sure you run the app once and select 'obama1' to extract its .wav file, or copy it manually.")
         sys.exit(1)

    print(f"Loading source: {target_audio.name}")
    y, sr = librosa.load(target_audio, sr=None)
    
    # 1. High Pitch (n_steps=6) -> Tests if stress/pitch penalty is correctly applied
    out_pitch = ref_audio_dir / "obama1_test_high_pitch.wav"
    print("Generating high pitch version...")
    y_pitch = shift_pitch(y, sr, n_steps=6)
    sf.write(out_pitch, y_pitch, sr)
    
    # 2. Fast Speed (1.8x) -> Tests if WPM penalty is correctly applied
    out_speed = ref_audio_dir / "obama1_test_fast.wav"
    print("Generating fast speed version...")
    y_fast = change_speed(y, speed_factor=1.8)
    sf.write(out_speed, y_fast, sr)

    # 3. Noisy (0.02) -> Tests if Energy/Volume penalty handles noise correctly
    out_noise = ref_audio_dir / "obama1_test_noisy.wav"
    print("Generating noisy version...")
    y_noise = add_noise(y, noise_level=0.02)
    sf.write(out_noise, y_noise, sr)

    print("Done! Test audio files created successfully in assets/reference_audio/")

if __name__ == "__main__":
    generate_augmented_audio()
