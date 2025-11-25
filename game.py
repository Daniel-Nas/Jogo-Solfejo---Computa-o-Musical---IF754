import pygame
import numpy as np
import sounddevice as sd
import threading
import time

# === Solfejo - jogo educativo de identificação de músicas por notas ===
# Requisitos: pygame, sounddevice, librosa, numpy
# Execute com o Python 3.11 (onde as dependências foram instaladas).

# --------------------------------------
# Configurações
# --------------------------------------
SAMPLE_RATE = 44100
LISTEN_DURATION = 1.5  # segundos para capturar a voz do jogador
SEMITONE_TOLERANCE = 0.6  # tolerância em semitons para considerar "acertou"

# --------------------------------------
# Inicialização Pygame
# --------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Solfejo - Jogo de Solfejo")
FONT = pygame.font.SysFont("arial", 24)
BIG = pygame.font.SysFont("arial", 34)
SMALL = pygame.font.SysFont("arial", 18)
CLOCK = pygame.time.Clock()

WHITE = (255, 255, 255)
DARK = (30, 30, 30)
ACCENT = (70, 140, 255)
GREEN = (100, 220, 100)
RED = (255, 90, 90)
YELLOW = (255, 220, 80)

# --------------------------------------
# Notas e frequências (temperamento igual, A4=440Hz)
# --------------------------------------
NOTAS_NAME = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_FREQS = {
    "C": 261.63, "C#": 277.18, "D": 293.66,
    "D#": 311.13, "E": 329.63, "F": 349.23,
    "F#": 369.99, "G": 392.00, "G#": 415.30,
    "A": 440.00, "A#": 466.16, "B": 493.88
}

# --------------------------------------
# Base de "músicas" (cada música é sequência de notas)
# --------------------------------------
SONGS = {
    "Brilha Brilha (tom Ré)": ["D", "D", "A", "A", "B", "B", "A"],
    "Parabéns Pra Você": ["C", "C", "D", "C", "F", "E"],
    "Cai Cai Balão": ["C", "C", "G", "G", "A", "A", "G"]
}

# --------------------------------------
# Botão simples
# --------------------------------------
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

# --------------------------------------
# Áudio: geração de nota simples (seno) e reprodução - agora retorna buffer pra análise
# Adicionado: parâmetro record para controlar se devemos salvar na lista played_notes
# --------------------------------------
played_notes = []  # lista de frequências (float) que foram efetivamente tocadas (ordem)

def synth_note(freq, duration=0.8, volume=0.5):
    """
    Retorna (stereo_int16, mono_float32_normalized)
    stereo_int16 -> para pygame.sndarray
    mono_float32_normalized -> para análise (valores em [-1,1])
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = (np.sin(2*np.pi*freq*t) * 32767 * volume).astype(np.int16)
    stereo = np.column_stack((wave, wave))
    mono_float = wave.astype(np.float32) / 32767.0
    return stereo, mono_float


def play_note(freq, duration=0.8, wait=False, record=True):
    """
    Reproduz a nota via pygame (sem retornar análise).
    se record=False não adiciona a freq em played_notes (usado para replays).
    """
    global currently_playing
    currently_playing = True
    if record:
        played_notes.append(float(freq))
    stereo_buf, _ = synth_note(freq, duration)
    snd = pygame.sndarray.make_sound(stereo_buf)
    channel = snd.play()
    while channel.get_busy():
        pygame.time.wait(10)
    currently_playing = False


def play_note_and_analyze(freq, duration=0.8, record=True):
    """
    Reproduz a nota e também detecta a frequência diretamente do buffer sintetizado.
    Essa detecção é 100% confiável para saber o que o Pygame emitiu.
    O resultado fica em last_buffer_detected_freq.
    """
    global currently_playing, last_buffer_detected_freq
    currently_playing = True
    if record:
        played_notes.append(float(freq))
    stereo_buf, mono_float = synth_note(freq, duration)
    snd = pygame.sndarray.make_sound(stereo_buf)
    channel = snd.play()

    # Detectar pitch do buffer sintetizado (imediato)
    detected = detect_pitch_autocorr(mono_float, SAMPLE_RATE)
    last_buffer_detected_freq = detected

    while channel.get_busy():
        pygame.time.wait(10)
    currently_playing = False

# --------------------------------------
# Funções de pitch detection (autocorrelação + interpolação)
# aplicáveis tanto para buffer sintetizado quanto para áudio gravado
# --------------------------------------
def parabolic_interpolation(y, x):
    """Refina pico para precisão sub-amostral."""
    if x <= 0 or x >= (len(y)-1):
        return x, y[x]
    alpha = y[x-1]
    beta  = y[x]
    gamma = y[x+1]
    p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
    xp = x + p
    yp = beta - 0.25 * (alpha - gamma) * p
    return xp, yp

def detect_pitch_autocorr(audio, sr):
    """
    Detecta frequência fundamental de um buffer mono (-1..1).
    Retorna freq em Hz (float) ou None.
    """
    x = np.array(audio, dtype=np.float64)
    if x.size == 0:
        return None
    x = x - np.mean(x)
    m = np.max(np.abs(x))
    if m < 1e-6:
        return None
    x = x / m

    # janela para reduzir leakage
    w = np.hanning(len(x))
    xw = x * w

    # autocorrelação via FFT (rápida)
    n = len(xw)
    # nfft: power of 2 sufficiently large
    nfft = 1 << ((2*n-1).bit_length())
    X = np.fft.rfft(xw, n=nfft)
    S = np.abs(X)**2
    corr = np.fft.irfft(S, n=nfft)[:n]

    # normalizar (evita overflow)
    maxcorr = np.max(np.abs(corr))
    if maxcorr == 0:
        return None
    corr = corr / maxcorr

    # encontrar primeiro ponto onde a derivada vira positiva (após zero-lag)
    d = np.diff(corr)
    starts = np.where(d > 0)[0]
    if starts.size == 0:
        return None
    start = int(starts[0])

    # encontrar pico máximo após start
    peak = start + int(np.argmax(corr[start:]))
    if peak <= 0:
        return None

    # interpolação parabólica para sub-amostra
    peak_x, _ = parabolic_interpolation(corr, peak)
    if peak_x == 0:
        return None

    freq = sr / peak_x
    if not np.isfinite(freq) or freq <= 0:
        return None
    return float(freq)

# Detector que processa áudio gravado (microfone/loopback) em janelas e retorna mediana
def detect_pitch_from_audio(audio, sr):
    a = np.array(audio, dtype=np.float64).flatten()
    if a.size == 0:
        return None
    # limpar
    a = np.nan_to_num(a)
    a -= np.mean(a)
    mx = np.max(np.abs(a))
    if mx < 1e-6:
        return None
    a /= mx

    hop = 1024
    block = 2048
    freqs = []
    for start in range(0, len(a) - block + 1, hop):
        blk = a[start:start+block]
        f = detect_pitch_autocorr(blk, sr)
        if f is not None and 50 < f < 5000:
            freqs.append(f)
    if len(freqs) == 0:
        return None
    return float(np.median(freqs))

# --------------------------------------
# Áudio: gravação
# --------------------------------------
def record_audio(duration=LISTEN_DURATION, sample_rate=SAMPLE_RATE):
    rec = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return rec.flatten(), sample_rate

# (opcional) tenta encontrar device loopback no Windows; retorna device index ou None
def find_loopback_device():
    try:
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            name = d['name'].lower()
            if 'loopback' in name or 'stereo mix' in name or 'wasapi' in d.get('hostapi', '').lower():
                return i
    except Exception:
        pass
    return None

def record_loopback(duration=LISTEN_DURATION, sr=SAMPLE_RATE):
    dev = find_loopback_device()
    if dev is None:
        return record_audio(duration, sr)
    rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32', device=dev)
    sd.wait()
    return rec.flatten(), sr

# --------------------------------------
# Funções utilitárias de nota
# --------------------------------------
def freq_to_note_name(freq):
    if freq is None or freq <= 0:
        return None
    n = 12 * np.log2(freq / 440.0)
    midi = int(round(n)) + 69
    name = NOTAS_NAME[midi % 12]
    return name

def is_pitch_match(detected_freq, target_freq):
    """
    Retorna True se a nota detectada possui o mesmo nome que a nota alvo,
    ignorando oitavas (por exemplo: C3, C4, C5 são todos "C").
    """
    if detected_freq is None or target_freq is None:
        return False

    detected_name = freq_to_note_name(detected_freq)
    target_name = freq_to_note_name(target_freq)

    return detected_name == target_name

# --------------------------------------
# Estado do jogo
# --------------------------------------
currently_playing = False
last_buffer_detected_freq = None  # freq detectada do buffer sintetizado (debug)
detected_name = None
detected_freq = None
detector_result = None

state = "menu"  # menu, rules, settings, play, detector, gameover
lives = 3
score = 0
current_song_name = None
current_song_seq = []
current_index = 0

# elementos UI
btn_start = Button("INICIAR", (WIDTH//2 - 140, 180, 280, 60))
btn_rules = Button("REGRAS", (WIDTH//2 - 140, 260, 280, 60))
btn_conf = Button("CONFIGURAÇÕES", (WIDTH//2 - 140, 340, 280, 60))

btn_back = Button("VOLTAR", (20, HEIGHT-70, 120, 50))

# Tela de detector: botões
btn_play_target = Button("Ouvir Nota", (WIDTH-220, 120, 180, 50), color=YELLOW)
btn_start_listen = Button("Cantar (Gravar)", (WIDTH-220, 190, 180, 50), color=GREEN)
btn_skip_confirm = Button("Liberar Próxima", (WIDTH-220, 260, 180, 50), color=ACCENT)

# Tela de jogo: ações
btn_repeat = Button("Repetir Notas", (50, 240, 280, 60), color=(120,160,220))
btn_skip = Button("SKIP (liberar próxima)", (50, 320, 280, 60), color=(200,120,120))
btn_guess = Button("TENTAR ADVINHAR", (50, 400, 280, 60), color=(120,180,200))


# input de texto para palpite
user_text = ""
input_active = False
message = ""

# escolhe uma música aleatória
def start_round():
    global current_song_name, current_song_seq, current_index, message, played_notes
    current_song_name = random_choice_from_dict(SONGS)
    current_song_seq = SONGS[current_song_name].copy()
    current_index = 0
    played_notes = []  # reinicia histórico de notas tocadas
    message = "Ouça a primeira nota ou tente advinhar a música."

def random_choice_from_dict(d):
    names = list(d.keys())
    return names[np.random.randint(0, len(names))]

# --------------------------------------
# Lógica para liberar próxima nota via solfejo
# --------------------------------------
def detector_listen_and_check(target_note):
    """
    Grava o áudio do jogador, detecta pitch e retorna o nome detectado e se bate com target_note.
    Essa função é bloqueante — chamamos em thread para não travar a UI.
    """
    global message, detected_name, detected_freq, detector_result
    detected_name = None
    detected_freq = None
    detector_result = None

    # garantir que não estamos tocando som do jogo (evitar captura de vazamento)
    while currently_playing:
        time.sleep(0.01)

    # grava áudio do microfone (ou loopback se disponível)
    audio, sr = record_audio()
    f = detect_pitch_from_audio(audio, sr)
    detected_freq = f
    detected_name = freq_to_note_name(f)
    if is_pitch_match(f, NOTE_FREQS[target_note]):
        detector_result = True
        message = f"Acertou! Detectado {detected_name} ({detected_freq:.1f} Hz)."
    else:
        detector_result = False
        if detected_name:
            message = f"Detectado {detected_name} ({detected_freq:.1f} Hz). Não bate com {target_note}."
        else:
            message = "Não foi possível detectar pitch. Tente novamente."

# thread wrapper
def start_detector_thread(target_note):
    t = threading.Thread(target=detector_listen_and_check, args=(target_note,), daemon=True)
    t.start()

# --------------------------------------
# Funções de desenho de telas
# --------------------------------------
def draw_menu():
    screen.fill(DARK)
    screen.blit(BIG.render("SOLFEJO", True, ACCENT), (WIDTH//2 - 100, 80))
    btn_start.draw(screen)
    btn_rules.draw(screen)
    btn_conf.draw(screen)
    screen.blit(SMALL.render("Versão educativa - Solfejo", True, WHITE), (WIDTH//2 - 110, 520))

def draw_rules():
    screen.fill(DARK)
    y = 60
    screen.blit(BIG.render("REGRAS", True, ACCENT), (40, 20))
    rules = [
        "- Você tem 3 vidas.",
        "- A cada rodada uma música é escolhida.",
        "- O jogo toca a primeira nota.",
        "- Você pode tentar advinhar a música; erro = -1 vida.",
        "- Ou pressionar SKIP para liberar a próxima nota:",
        "    para liberar, você precisa cantar a nota atual no detector.",
        "- Ao ouvir a nota (botão 'Ouvir Nota') o detector fica inativo até a reprodução terminar.",
    ]
    for r in rules:
        screen.blit(FONT.render(r, True, WHITE), (40, y))
        y += 34
    btn_back.draw(screen)

def draw_settings():
    screen.fill(DARK)
    screen.blit(BIG.render("CONFIGURAÇÕES", True, ACCENT), (40, 20))
    screen.blit(FONT.render(f"Tolerância (semitons): {SEMITONE_TOLERANCE}", True, WHITE), (40, 80))
    btn_back.draw(screen)

def draw_play():
    screen.fill(DARK)
    # Status
    screen.blit(FONT.render(f"Vidas: {lives}", True, WHITE), (40, 20))
    screen.blit(FONT.render(f"Pontos: {score}", True, WHITE), (200, 20))
    screen.blit(FONT.render(f"Música: {current_song_name or '---'}", True, WHITE), (40, 60))
    # Mostra notas liberadas
    displayed = current_song_seq[:current_index]
    txt = " ".join(displayed) if displayed else "(nenhuma)"
    screen.blit(FONT.render(f"Notas liberadas: {txt}", True, WHITE), (40, 100))

    # botões principais
    btn_repeat.draw(screen)
    btn_skip.draw(screen)
    btn_guess.draw(screen)

    # texto de mensagem
    screen.blit(SMALL.render(message, True, YELLOW), (40, 480))

    # se houver nota atual, mostra botão de reproduzir
    if current_index < len(current_song_seq):
        cur_note = current_song_seq[current_index]
        screen.blit(FONT.render(f"Nota atual: {cur_note}", True, WHITE), (40, 140))
        # botão para tocar nota: ao tocar, set currently_playing True
        btn_play_here = Button("Ouvir Nota Atual", (40, 170, 200, 40), color=YELLOW)
        btn_play_here.draw(screen)
        # salvar para ver clique
        global play_here_button
        play_here_button = btn_play_here

    # mostra a frequência detectada do buffer sintetizado para depuração
    if last_buffer_detected_freq:
        s = FONT.render(f"Buffer freq: {last_buffer_detected_freq:.2f} Hz", True, (200,200,200))
        screen.blit(s, (260, 170))

    # campo de input para chute
    pygame.draw.rect(screen, (50,50,50), (370, 350, 420, 40), border_radius=6)
    display_text = user_text if (input_active or user_text) else "Digite o nome da música e pressione TENTAR"
    screen.blit(FONT.render(display_text, True, WHITE), (380, 358))

def draw_detector():
    screen.fill(DARK)
    screen.blit(BIG.render("DETECTOR DE PITCH", True, ACCENT), (40, 20))
    # proteger índice caso o usuário acesse detector sem nota
    if current_index < len(current_song_seq):
        screen.blit(FONT.render(f"Nota alvo: {current_song_seq[current_index]}", True, WHITE), (40, 90))
    else:
        screen.blit(FONT.render("Nota alvo: ---", True, WHITE), (40, 90))

    # botões do detector
    btn_play_target.draw(screen)
    btn_start_listen.draw(screen)
    btn_skip_confirm.draw(screen)

    # exibir leitura detectada
    det = globals().get('detected_name', None)
    detf = globals().get('detected_freq', None)
    if det:
        screen.blit(FONT.render(f"Detectado: {det} ({detf:.1f} Hz)", True, GREEN), (40, 180))

    # resultado
    res = globals().get('detector_result', None)
    if res is True:
        screen.blit(FONT.render("Detector: ACERTO! Próxima nota liberada.", True, GREEN), (40, 220))
    elif res is False:
        screen.blit(FONT.render("Detector: NÃO bate. Tente de novo.", True, RED), (40, 220))

    btn_back.draw(screen)

# --------------------------------------
# Loop principal
# --------------------------------------
running = True
start_time = time.time()

# garantir que play_here_button existe
play_here_button = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if state == 'menu':
            if btn_start.clicked(event):
                # iniciar rodada
                lives = 3
                score = 0
                start_round()
                # tocar primeira nota (usa play_note_and_analyze para registrar buffer freq)
                current = current_song_seq[0]
                threading.Thread(target=play_note_and_analyze, args=(NOTE_FREQS[current], 0.8), daemon=True).start()
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
                # toca todas as notas já tocadas (histórico). Não registra novamente (record=False)
                def play_released_sequence_from_history():
                    seq = list(played_notes)  # cópia
                    for freq in seq:
                        play_note(freq, 0.6, record=False)
                        time.sleep(0.05)
                threading.Thread(target=play_released_sequence_from_history, daemon=True).start()

            if play_here_button and play_here_button.clicked(event):
                # tocar nota atual (e analisar buffer)
                if current_index < len(current_song_seq):
                    cur = current_song_seq[current_index]
                    threading.Thread(target=play_note_and_analyze, args=(NOTE_FREQS[cur], 0.8), daemon=True).start()

            if btn_skip.clicked(event):
                # abrir detector para liberar próxima nota
                state = 'detector'
                # reset detector state
                globals().pop('detected_name', None)
                globals().pop('detected_freq', None)
                globals().pop('detector_result', None)

            if btn_guess.clicked(event):
                # ativar input
                input_active = True

            # captura teclado para input se ativo
            if event.type == pygame.KEYDOWN and input_active:
                if event.key == pygame.K_RETURN:
                    # avaliar chute
                    guess = user_text.strip().lower()
                    if guess == (current_song_name or "").lower():
                        score += 5
                        message = f"Parabéns! Você acertou: {current_song_name}."
                        # iniciar nova rodada
                        start_round()
                        threading.Thread(target=play_note_and_analyze, args=(NOTE_FREQS[current_song_seq[0]], 0.8), daemon=True).start()
                        input_active = False
                        user_text = ""
                    else:
                        lives -= 1
                        message = f"Errado! Você perdeu 1 vida. Vidas restantes: {lives}"
                        user_text = ""
                        input_active = False
                        if lives <= 0:
                            state = 'gameover'
                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    # aceitar apenas caracteres
                    if len(user_text) < 60 and event.unicode.isprintable():
                        user_text += event.unicode

        elif state == 'detector':
            if btn_back.clicked(event):
                state = 'play'
            if btn_play_target.clicked(event):
                # reproduz a nota alvo; durante reprodução detector fica inativo
                if current_index < len(current_song_seq):
                    cur = current_song_seq[current_index]
                    threading.Thread(target=play_note_and_analyze, args=(NOTE_FREQS[cur], 0.8), daemon=True).start()
            if btn_start_listen.clicked(event):
                # inicia gravação e detecção em thread
                if current_index < len(current_song_seq):
                    target = current_song_seq[current_index]
                    # start detector thread (bloqueante internamente, mas não bloqueia UI)
                    start_detector_thread(target)
            if btn_skip_confirm.clicked(event):
                # somente libera a próxima nota se detector_result for True
                if globals().get('detector_result', None) is True:
                    # libera próxima nota
                    current_index += 1
                    message = f"Próxima nota liberada: {current_song_seq[:current_index]}"
                    globals().pop('detector_result', None)
                    globals().pop('detected_name', None)
                    globals().pop('detected_freq', None)
                    state = 'play'
                    # tocar as notas liberadas (pequena sequência para feedback) sem duplicar histórico
                    def play_released_sequence():
                        # se existem played_notes (notas reais tocadas), prefira reproduzir essas;
                        # caso contrário, toque a sequência nominal até current_index
                        if len(played_notes) >= current_index:
                            seq = played_notes[:current_index]
                        else:
                            seq = [NOTE_FREQS[n] for n in current_song_seq[:current_index]]
                        for freq in seq:
                            play_note(freq, 0.6, record=False)
                            time.sleep(0.05)
                    threading.Thread(target=play_released_sequence, daemon=True).start()
                else:
                    message = "Você precisa cantar corretamente antes de liberar."

    # desenha telas
    if state == 'menu':
        draw_menu()
    elif state == 'rules':
        draw_rules()
    elif state == 'settings':
        draw_settings()
    elif state == 'play':
        draw_play()
    elif state == 'detector':
        draw_detector()
    elif state == 'gameover':
        screen.fill(DARK)
        screen.blit(BIG.render("GAME OVER", True, RED), (WIDTH//2 - 120, HEIGHT//2 - 60))
        screen.blit(FONT.render(f"Pontos: {score}", True, WHITE), (WIDTH//2 - 40, HEIGHT//2))
        btn_back.draw(screen)

    pygame.display.flip()
    CLOCK.tick(30)

pygame.quit()
