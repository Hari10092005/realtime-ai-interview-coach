import sounddevice as sd
import numpy as np

fs = 44100  # sample rate
duration = 5  # seconds

print("🎤 Speak now...")

# Record audio
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

print("Processing...")

# Convert to numpy array
audio = audio.flatten()

# Feature 1: Volume (energy)
volume = np.mean(np.abs(audio))

# Feature 2: Variation (stress indicator)
variation = np.std(audio)

# Feature 3: Peak (sudden stress)
peak = np.max(np.abs(audio))

# -----------------------
# SIMPLE AI LOGIC
# -----------------------
if volume > 0.1 and variation > 0.05:
    result = "Stressed 😰"
elif volume > 0.05:
    result = "Confident 😎"
else:
    result = "Nervous 😬"

print(f"\n🎯 Voice Analysis Result: {result}")
print(f"Volume: {volume:.4f}, Variation: {variation:.4f}, Peak: {peak:.4f}")