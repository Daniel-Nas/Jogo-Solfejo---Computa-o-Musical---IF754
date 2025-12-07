import pygame
import numpy as np
import threading
import time
import pyaudio
import aubio
import math
import random

# IMPORTAÇÃO DA NOVA ESTRUTURA
from Musicas import BIBLIOTECA, Musica

# ==============================================================================
# CONFIGURAÇÕES GERAIS
# ==============================================================================
SAMPLE_RATE = 44100     # Padrão mais seguro
LISTEN_DURATION = 10.0  
REQUIRED_STABILITY = 1.0 
A4_TUNING = 440.0       

# AJUSTE FINO DE AFINAÇÃO
TUNING_OFFSET = 0  
TUNING_MULTIPLIER = 2 ** (TUNING_OFFSET / 12.0)

# ==============================================================================
# 1. CLASSE PITCH DETECTOR
# ==============================================================================
class PitchDetector:
    def __init__(self):
        self.BUFFER_SIZE = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100 
        self.A4 = A4_TUNING
        self.NOTAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.running = False
        self.current_note = None
        self.current_freq = 0.0
        self._thread = None

    def _freq_para_nota(self, freq):
        if freq <= 0: return None
        try:
            n = 12 * math.log2(freq / self.A4) + 69
            n_round = int(round(n))
            nome = self.NOTAS[n_round % 12]
            oitava = (n_round // 12) - 1
            return f"{nome}{oitava}"
        except ValueError:
            return None

    def _listen_loop(self):
        p = pyaudio.PyAudio()
        pitch_detector = aubio.pitch("default", self.BUFFER_SIZE*4, self.BUFFER_SIZE, self.RATE)
        pitch_detector.set_unit("Hz")
        pitch_detector.set_silence(-40)

        try:
            stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.BUFFER_SIZE)
            while self.running:
                try:
                    audio_data = stream.read(self.BUFFER_SIZE, exception_on_overflow=False)
                    samples = np.frombuffer(audio_data, dtype=np.float32)
                    freq = pitch_detector(samples)[0]
                    self.current_freq = float(freq)
                    self.current_note = self._freq_para_nota(freq)
                except Exception:
                    pass
        except Exception as e:
            print(f"Erro no detector: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            p.terminate()

    def start(self):
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.current_note = None
        self.current_freq = 0.0

    def get_note(self): return self.current_note
    def get_freq(self): return self.current_freq

detector = PitchDetector()

# ==============================================================================
# 2. INICIALIZAÇÃO E UI
# ==============================================================================
# Aumentei o buffer para 4096 para evitar "estalos" (crackling)
pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=4096)
pygame.init()

WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Solfejo - Piano Suave")

# Cores
WHITE = (255, 255, 255)
DARK = (30, 30, 30)
ACCENT = (70, 140, 255)
GREEN = (100, 220, 100)
RED = (255, 90, 90)
YELLOW = (255, 220, 80)

FONT = pygame.font.SysFont("arial", 24)
BIG = pygame.font.SysFont("arial", 34)
SMALL = pygame.font.SysFont("arial", 18)
CLOCK = pygame.time.Clock()

# Tabela de Frequências Base
NOTAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_FREQS = {}
for i, nome in enumerate(NOTAS):
    distancia = i - 9
    freq = A4_TUNING * (2 ** (distancia / 12.0))
    NOTE_FREQS[nome] = freq

# ==============================================================================
# 3. SINTETIZADOR DE PIANO CORRIGIDO (LIMITER + VOLUME BAIXO)
# ==============================================================================
class Button:
    def __init__(self, text, rect, color=ACCENT, hover=None):
        self.text = text
        self.rect = pygame.Rect(rect)
        self.color = color
        self.hover = hover or tuple(min(255, c+30) for c in color)

    def draw(self, surf):
        m = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(m)
        col = self.hover if is_hover else self.color
        pygame.draw.rect(surf, col, self.rect, border_radius=8)
        s = FONT.render(self.text, True, WHITE)
        surf.blit(s, (self.rect.x + (self.rect.w - s.get_width())//2,
                      self.rect.y + (self.rect.h - s.get_height())//2))

    def clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

played_notes = [] 
currently_playing = False

def synth_piano_note(base_freq, duration=1.0, volume=0.3): # Volume padrão reduzido para 0.3
    """
    Gera som de piano elétrico com proteção contra distorção (Clipping).
    """
    if base_freq <= 0: return None
    
    # Aplica correção de afinação
    freq = base_freq * TUNING_MULTIPLIER
    
    length = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, length, False)
    
    # 1. Fundamental
    wave = np.sin(2 * np.pi * freq * t)
    
    # 2. Harmônicos (Reduzidos para evitar sobrecarga)
    wave += 0.4 * np.sin(2 * np.pi * (freq * 2) * t) 
    wave += 0.1 * np.sin(2 * np.pi * (freq * 3) * t)
    
    # 3. Envelope (Decay Suave)
    decay = np.exp(-t * 3) 
    wave *= decay
    
    # 4. NORMALIZAÇÃO E LIMITER (O SEGREDO PARA NÃO ESTOURAR)
    # Primeiro, normaliza para o maior pico ser 1.0
    max_val = np.max(np.abs(wave))
    if max_val > 0:
        wave = wave / max_val
    
    # Aplica o volume desejado
    wave = wave * volume
    
    # CLAMP: Garante que NENHUM número passe de 0.99 ou -0.99
    # Isso impede a distorção digital (clipping)
    wave = np.clip(wave, -0.99, 0.99)
    
    # 5. Converte para 16-bit
    wave = (wave * 32767).astype(np.int16)
    
    stereo = np.column_stack((wave, wave))
    return stereo

def play_note(freq, duration, record=True):
    global currently_playing
    currently_playing = True
    
    if record:
        played_notes.append((float(freq), duration))
    
    try:
        stereo_buf = synth_piano_note(freq, duration)
        
        if stereo_buf is not None:
            snd = pygame.sndarray.make_sound(stereo_buf)
            channel = snd.play()
            
            # Fadeout suave se a nota for longa
            start_t = time.time()
            while channel.get_busy() and (time.time() - start_t) < duration:
                pygame.time.wait(10)
            
            if (time.time() - start_t) >= duration:
                channel.fadeout(50)
        else:
            time.sleep(duration)
            
    except Exception as e:
        print(f"Erro audio: {e}")
        time.sleep(duration)
    
    time.sleep(0.05)
    currently_playing = False


# ==============================================================================
# 4. LÓGICA DO DETECTOR
# ==============================================================================
def detector_process(target_note_name):
    global message, detected_name, detected_freq, detector_result

    detected_name = None
    detected_freq = None
    detector_result = None

    while currently_playing:
        time.sleep(0.01)

    detector.start()
    message = "Prepare-se... Cante e SEGURE a nota!"
    
    session_start_time = time.time()
    stable_start_time = None 
    found_match = False
    
    while time.time() - session_start_time < LISTEN_DURATION:
        note_completa = detector.get_note()
        freq = detector.get_freq()
        
        if note_completa:
            detected_name = note_completa
            detected_freq = freq
            note_only_name = ''.join([c for c in note_completa if not c.isdigit()])
            
            if note_only_name == target_note_name:
                if stable_start_time is None:
                    stable_start_time = time.time()
                
                elapsed = time.time() - stable_start_time
                message = f"SEGURE! {elapsed:.1f}s / {REQUIRED_STABILITY}s"
                
                if elapsed >= REQUIRED_STABILITY:
                    detector_result = True
                    message = f"ACERTOU! Nota {detected_name} confirmada."
                    found_match = True
                    break 
            else:
                stable_start_time = None
                message = f"Detectado: {detected_name}. Buscando: {target_note_name}"
        else:
            stable_start_time = None
            message = "Silêncio..."
        
        time.sleep(0.05)

    detector.stop()

    if not found_match:
        detector_result = False
        message = "Tempo esgotado."

def start_detector_thread(target_note):
    t = threading.Thread(target=detector_process, args=(target_note,), daemon=True)
    t.start()

# ==============================================================================
# 5. UI E LOOP
# ==============================================================================
state = "menu"
lives = 3
score = 0

current_song_data = None    
current_song_seq = []       
current_index = 0

message = ""
user_text = ""
input_active = False
currently_playing = False
detected_name = None
detected_freq = None
detector_result = None

btn_start = Button("INICIAR", (WIDTH//2 - 140, 180, 280, 60))
btn_rules = Button("REGRAS", (WIDTH//2 - 140, 260, 280, 60))
btn_conf = Button("CONFIGURAÇÕES", (WIDTH//2 - 140, 340, 280, 60))
btn_back = Button("VOLTAR", (20, HEIGHT-70, 120, 50))

btn_repeat = Button("Repetir Notas", (50, 240, 280, 60), color=(120,160,220))
btn_action_sing = Button("🎤 CANTAR NOTA", (50, 320, 280, 60), color=(255, 140, 100))
btn_guess = Button("TENTAR ADVINHAR", (50, 400, 280, 60), color=(120,180,200))

btn_play_target = Button("Ouvir Nota Alvo", (WIDTH-220, 120, 180, 50), color=YELLOW)
btn_start_listen = Button("Gravar (Microfone)", (WIDTH-220, 190, 180, 50), color=GREEN)
btn_skip_confirm = Button("Confirmar e Voltar", (WIDTH-220, 260, 180, 50), color=ACCENT)


def start_round():
    global current_song_data, current_song_seq, current_index, message, played_notes
    current_song_data = random.choice(BIBLIOTECA)
    current_song_seq = current_song_data.notas 
    current_index = 0
    played_notes = []
    message = "Ouça a primeira nota ou tente advinhar a música."

def draw_menu():
    screen.fill(DARK)
    screen.blit(BIG.render("SOLFEJO", True, ACCENT), (WIDTH//2 - 100, 80))
    btn_start.draw(screen)
    btn_rules.draw(screen)
    btn_conf.draw(screen)
    screen.blit(SMALL.render("Piano Suave (Anti-Clipping)", True, WHITE), (WIDTH//2 - 130, 520))

def draw_rules():
    screen.fill(DARK)
    y = 60
    screen.blit(BIG.render("REGRAS", True, ACCENT), (40, 20))
    rules = [
        "- Você tem 3 vidas.",
        "- Se não souber a próxima nota, clique em CANTAR NOTA.",
        "- Para desbloquear, CANTE e SEGURE a nota por 1 segundo.",
        "- Adivinhe a música digitando o nome."
    ]
    for r in rules:
        screen.blit(FONT.render(r, True, WHITE), (40, y))
        y += 34
    btn_back.draw(screen)

def draw_settings():
    screen.fill(DARK)
    screen.blit(BIG.render("CONFIGURAÇÕES", True, ACCENT), (40, 20))
    screen.blit(FONT.render(f"Offset: +{TUNING_OFFSET} Semitons", True, WHITE), (40, 80))
    btn_back.draw(screen)

def draw_play():
    screen.fill(DARK)
    screen.blit(FONT.render(f"Vidas: {lives}", True, WHITE), (40, 20))
    screen.blit(FONT.render(f"Pontos: {score}", True, WHITE), (200, 20))
    
    nome_show = current_song_data.nome if state == 'gameover' else '---'
    screen.blit(FONT.render(f"Música: {nome_show}", True, WHITE), (40, 60))
    
    displayed = [n[0] for n in current_song_seq[:current_index]]
    txt = " ".join(displayed) if displayed else "(nenhuma)"
    screen.blit(FONT.render(f"Notas liberadas: {txt}", True, WHITE), (40, 100))

    btn_repeat.draw(screen)
    btn_action_sing.draw(screen)
    btn_guess.draw(screen)
    screen.blit(SMALL.render(message, True, YELLOW), (40, 480))

    global play_here_button
    if current_index < len(current_song_seq):
        screen.blit(FONT.render(f"Próxima nota: ???", True, WHITE), (40, 140))
        btn_play_here = Button("Ouvir Nota Atual", (40, 170, 200, 40), color=YELLOW)
        btn_play_here.draw(screen)
        play_here_button = btn_play_here

    pygame.draw.rect(screen, (50,50,50), (370, 350, 420, 40), border_radius=6)
    display_text = user_text if (input_active or user_text) else "Digite o nome da música..."
    col = WHITE if input_active else (150,150,150)
    screen.blit(FONT.render(display_text, True, col), (380, 358))

def draw_detector():
    screen.fill(DARK)
    screen.blit(BIG.render("DETECTOR DE PITCH", True, ACCENT), (40, 20))
    target = current_song_seq[current_index][0] if current_index < len(current_song_seq) else "-"
    screen.blit(FONT.render(f"Cante e SEGURE a nota: {target}", True, WHITE), (40, 90))
    btn_play_target.draw(screen)
    btn_start_listen.draw(screen)
    btn_skip_confirm.draw(screen)
    if detected_name:
        screen.blit(FONT.render(f"Detectado: {detected_name} ({detected_freq:.1f} Hz)", True, GREEN), (40, 180))
    msg_color = YELLOW if detector.running else WHITE
    if detector_result is True: msg_color = GREEN
    elif detector_result is False: msg_color = RED
    screen.blit(FONT.render(message, True, msg_color), (40, 230))
    btn_back.draw(screen)

# ==============================================================================
# LOOP PRINCIPAL
# ==============================================================================
running = True
play_here_button = None 

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if state == 'menu':
            if btn_start.clicked(event):
                lives = 3
                score = 0
                start_round()
                if current_song_seq:
                    primeira_nota = current_song_seq[0]
                    threading.Thread(target=play_note, args=(NOTE_FREQS[primeira_nota[0]], primeira_nota[1]), daemon=True).start()
                state = 'play'
            if btn_rules.clicked(event):
                state = 'rules'
            if btn_conf.clicked(event):
                state = 'settings'

        elif state in ('rules', 'settings'):
            if btn_back.clicked(event):
                state = 'menu'

        elif state == 'play':
            if btn_repeat.clicked(event):
                def replay():
                    seq = list(played_notes)
                    for freq, dur in seq:
                        play_note(freq, dur, record=False)
                threading.Thread(target=replay, daemon=True).start()

            if play_here_button and play_here_button.clicked(event):
                if current_index < len(current_song_seq):
                    n = current_song_seq[current_index]
                    threading.Thread(target=play_note, args=(NOTE_FREQS[n[0]], n[1]), daemon=True).start()

            if btn_action_sing.clicked(event):
                state = 'detector'
                detector_result = None
                detected_name = None
                message = "Clique em Gravar e segure a nota por 1s."

            if btn_guess.clicked(event):
                input_active = True
                user_text = ""

            if event.type == pygame.KEYDOWN and input_active:
                if event.key == pygame.K_RETURN:
                    guess = user_text.strip().lower()
                    real = (current_song_data.nome or "").lower()
                    if guess in real and len(guess) > 3:
                        score += 5
                        message = f"ACERTOU: {current_song_data.nome}!"
                        start_round()
                        if current_song_seq:
                            n = current_song_seq[0]
                            threading.Thread(target=play_note, args=(NOTE_FREQS[n[0]], n[1]), daemon=True).start()
                    else:
                        lives -= 1
                        message = f"Errou! Vidas: {lives}"
                        if lives <= 0:
                            state = 'gameover'
                    user_text = ""
                    input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    if len(user_text) < 40: user_text += event.unicode

        elif state == 'detector':
            if btn_back.clicked(event):
                detector.stop()
                state = 'play'

            if btn_play_target.clicked(event):
                if current_index < len(current_song_seq):
                    n = current_song_seq[current_index]
                    threading.Thread(target=play_note, args=(NOTE_FREQS[n[0]], n[1]), daemon=True).start()

            if btn_start_listen.clicked(event):
                if current_index < len(current_song_seq):
                    target_name = current_song_seq[current_index][0]
                    start_detector_thread(target_name)

            if btn_skip_confirm.clicked(event):
                if detector_result is True:
                    current_index += 1
                    message = "Nota desbloqueada!"
                    state = 'play'
                    def play_released():
                        seq_to_play = []
                        if len(played_notes) >= current_index:
                            seq_to_play = played_notes[:current_index]
                            for freq, dur in seq_to_play:
                                play_note(freq, dur, record=False)
                        else:
                            seq_to_play = current_song_seq[:current_index]
                            for n in seq_to_play:
                                play_note(NOTE_FREQS[n[0]], n[1], record=False)
                    threading.Thread(target=play_released, daemon=True).start()
                else:
                    message = "Segure a nota por 1s até aparecer ACERTOU."

        elif state == 'gameover':
            if btn_back.clicked(event):
                state = 'menu'

    if state == 'menu': draw_menu()
    elif state == 'rules': draw_rules()
    elif state == 'settings': draw_settings()
    elif state == 'play': draw_play()
    elif state == 'detector': draw_detector()
    elif state == 'gameover':
        screen.fill(DARK)
        screen.blit(BIG.render("FIM DE JOGO", True, RED), (WIDTH//2 - 100, HEIGHT//2 - 50))
        screen.blit(FONT.render(f"Pontuação: {score}", True, WHITE), (WIDTH//2 - 60, HEIGHT//2))
        btn_back.draw(screen)

    pygame.display.flip()
    CLOCK.tick(30)

pygame.quit()