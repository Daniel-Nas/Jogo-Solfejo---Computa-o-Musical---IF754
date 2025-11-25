import aubio
import numpy as np
import pyaudio
import argparse
import time

# ==============================================================================
# CONFIGURAÇÕES DE CALIBRAGEM
# ==============================================================================
BUFFER_SIZE = 1024 
SAMPLE_RATE = 44100 # Se continuar ruim, tente mudar para 48000

# TRAVA 1: FILTRO DE FREQUÊNCIA (O Segredo!)
# Notas abaixo de 110Hz (Lá grave) ou acima de 800Hz (Sol agudo) serão ignoradas.
# Isso mata o ruído grave (70Hz) e o chiado agudo (1900Hz) que apareceu no seu print.
MIN_FREQ = 60 #Diminui pra ver se captura notas graves
MAX_FREQ = 800

# TRAVA 2: VOLUME
# Se a barrinha de volume passar do máximo, afaste o microfone.
VOLUME_THRESHOLD = 0.005 

parser = argparse.ArgumentParser()
parser.add_argument("-input", required=False, type=int, default=1, help="ID do Microfone")
args = parser.parse_args()

print(f"--- CALIBRADOR DE NOTAS (Filtro: {MIN_FREQ}Hz a {MAX_FREQ}Hz) ---")
print(f"Usando Mic ID: {args.input}")
print("OBSERVE A BARRA DE VOLUME. Se ficar vermelha (!!!), afaste o microfone.")

# ==============================================================================
# SETUP
# ==============================================================================
p = pyaudio.PyAudio()

try:
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1, rate=SAMPLE_RATE, input=True,
                    input_device_index=args.input, 
                    frames_per_buffer=BUFFER_SIZE)
except:
    print("Erro ao abrir microfone. Tente mudar o SAMPLE_RATE no código para 48000.")
    exit()

time.sleep(1)

# Usando algoritmo YIN com tolerância ajustada
pDetection = aubio.pitch("yin", 2048, BUFFER_SIZE, SAMPLE_RATE)
pDetection.set_unit("Hz")
pDetection.set_silence(-50) # Mais permissivo para captar, filtramos depois
pDetection.set_tolerance(0.8)

def freq_to_note(freq):
    NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    midi = 69 + 12 * np.log2(freq / 440.0)
    midi_round = int(round(midi))
    note = NOTES[midi_round % 12]
    octave = (midi_round // 12) - 1
    diff = midi - midi_round
    return f"{note}{octave}", diff

# Variáveis de estabilidade
last_note = ""
stable_count = 0

# ==============================================================================
# LOOP
# ==============================================================================
try:
    while True:
        data = stream.read(BUFFER_SIZE, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.float32)
        
        pitch = pDetection(samples)[0]
        confidence = pDetection.get_confidence() # Confiança do algoritmo (0.0 a 1.0)
        volume = np.sum(samples**2) / len(samples)

        # Barra de volume visual
        vol_bar = "|" * int(volume * 1000)
        if len(vol_bar) > 20: vol_bar = vol_bar[:20] + "!!!" # Alerta de estouro

        # Lógica de Filtro
        if volume > VOLUME_THRESHOLD:
            
            # FILTRO 1: Frequência absurda?
            if pitch < MIN_FREQ or pitch > MAX_FREQ:
                # Ignora silenciosamente (ruído de fundo)
                continue

            # FILTRO 2: Confiança baixa? (O algoritmo "acha" que é uma nota, mas não tem certeza)
            if confidence < 0.6: 
                continue

            note_data = freq_to_note(pitch)
            note_name, diff = note_data
            
            # Lógica de estabilidade (tem que repetir 2 vezes pra aparecer)
            if note_name == last_note:
                stable_count += 1
            else:
                stable_count = 0
                last_note = note_name

            if stable_count >= 1: # Se a nota se manteve por 2 frames
                print(f"Vol:[{vol_bar:<20}] Freq: {pitch:.1f}Hz \tNota: {note_name} \t(Confiança: {confidence:.2f})")
                stable_count = 0 # Reseta para não flodar a tela
                
        else:
            stable_count = 0

except KeyboardInterrupt:
    print("\nParando...")
    stream.stop_stream()
    stream.close()
    p.terminate()