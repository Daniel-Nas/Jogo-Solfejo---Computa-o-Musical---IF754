import sounddevice as sd
import numpy as np
import torch
import torchcrepe
from scipy.ndimage import median_filter
import math

# ------------------ CONFIGURAÇÕES ------------------ #
SAMPLE_RATE = 16000
HOP_LENGTH = 160   # 10 ms
WINDOW = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "full"  # "tiny" = rápido / "full" = precisão máxima

FMIN = 50
FMAX = 2000
PERIODICITY_THRESHOLD = 0.75  # quanto maior, mais preciso
SMOOTHING = 5  # suavização temporal (melhora precisão)
# --------------------------------------------------- #

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

def freq_to_midi(f):
    return 69 + 12 * math.log2(f / 440.0)

def midi_to_note_name(midi_value):
    rounded = int(round(midi_value))
    name = NOTE_NAMES[rounded % 12]
    octave = (rounded // 12) - 1
    return f"{name}{octave}"


print("\n=== Detector de Notas com IA (CREPE) ===")
print("Dispositivo:", DEVICE)
print("Fale uma nota... (Ctrl+C para sair)\n")

stream = sd.InputStream(
    channels=1, samplerate=SAMPLE_RATE, blocksize=WINDOW
)
stream.start()

try:
    audio_buffer = np.zeros(0, dtype=np.float32)

    while True:
        # leitura contínua
        audio_chunk, _ = stream.read(WINDOW)
        audio_chunk = audio_chunk.flatten().astype(np.float32)

        # concatena no buffer
        audio_buffer = np.concatenate([audio_buffer, audio_chunk])

        # limita buffer a 1 segundo
        if len(audio_buffer) > SAMPLE_RATE:
            audio_buffer = audio_buffer[-SAMPLE_RATE:]

        # normalização
        if np.max(np.abs(audio_buffer)) > 0:
            norm_audio = audio_buffer / np.max(np.abs(audio_buffer))
        else:
            continue

        # Tensor
        audio_tensor = torch.tensor(norm_audio).unsqueeze(0).to(DEVICE)

        # --------- PREDIÇÃO DO CREPE ---------- #
        freq, periodicity = torchcrepe.predict(
            audio_tensor,
            SAMPLE_RATE,
            HOP_LENGTH,
            FMIN,
            FMAX,
            MODEL,
            device=DEVICE,
            batch_size=1,
            return_periodicity=True
        )
        freq = freq.squeeze().cpu().numpy()
        periodicity = periodicity.squeeze().cpu().numpy()

        # filtra frames confiáveis
        good = periodicity > PERIODICITY_THRESHOLD
        if not np.any(good):
            print("Nota: --- (sem detecção confiável)", end="\r")
            continue

        # valores válidos
        valid_freq = freq[good]

        # suavização (melhora muito a precisão!)
        valid_freq = median_filter(valid_freq, size=SMOOTHING)

        f0 = np.mean(valid_freq)

        if f0 <= 0 or not np.isfinite(f0):
            print("Nota: ---", end="\r")
            continue

        midi = freq_to_midi(f0)
        note = midi_to_note_name(midi)

        print(f"Nota: {note:<4} | {f0:7.1f} Hz", end="\r")

except KeyboardInterrupt:
    print("\n\nFinalizado.")

stream.stop()
stream.close()
