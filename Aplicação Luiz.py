import pygame
import numpy as np
import random
import sounddevice as sd
import librosa

# --------------------------------------
# INICIALIZAÇÃO
# --------------------------------------
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

WIDTH, HEIGHT = 800, 550
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jogo de Notas - Luiz")

FONT = pygame.font.SysFont("arial", 30)
SMALL = pygame.font.SysFont("arial", 22)

WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
DARK_GRAY = (40, 40, 40)
BLUE = (70, 140, 255)
GREEN = (100, 220, 100)
RED = (255, 80, 80)
ORANGE = (255, 150, 70)
PURPLE = (180, 120, 255)

MUSICAS = {
    "Coração de Gelo": "music.mp3",
    "Sem Graça": "music2.mp3",
    "Além do que se vê": "music3.mp3"
}

# --------------------------------------
# BOTÕES
# --------------------------------------
class Button:
    def __init__(self, text, x, y, w, h, color, hover_color):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color
        self.hover_color = hover_color

    def draw(self):
        mouse = pygame.mouse.get_pos()
        color = self.hover_color if self.rect.collidepoint(mouse) else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=10)

        text_surface = FONT.render(self.text, True, WHITE)
        screen.blit(text_surface, (self.rect.x + (self.rect.width - text_surface.get_width()) // 2,
                                   self.rect.y + (self.rect.height - text_surface.get_height()) // 2))

    def clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)


# --------------------------------------
# FUNÇÃO PARA TOCAR NOTA
# --------------------------------------
def play_note(freq, duration=1.0, volume=0.5):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = (np.sin(freq * t * 2 * np.pi) * 32767 * volume).astype(np.int16)
    wave_stereo = np.column_stack((wave, wave))
    sound = pygame.sndarray.make_sound(wave_stereo)
    sound.play()
    pygame.time.delay(int(duration * 1000))


# --------------------------------------
# MICROFONE
# --------------------------------------
def record_audio(duration=1.5, sample_rate=44100):
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype='float32')
    sd.wait()
    return audio.flatten(), sample_rate


# --------------------------------------
# DETECÇÃO DE PITCH
# --------------------------------------
def detect_pitch(audio, sample_rate):
    audio = audio.astype(np.float32)
    f0 = librosa.yin(audio, fmin=50, fmax=1000)
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) > 0 else None


# --------------------------------------
# TABELA DE NOTAS
# --------------------------------------


def freq_to_note(freq):
    if freq <= 0:
        return None
    n = 12 * np.log2(freq / 440.0)
    midi = round(n) + 69
    note_name = NOTAS_NAME[midi % 12]
    return note_name


# --------------------------------------
# EXTRAIR NOTAS DA MÚSICA
# --------------------------------------
def extract_notes_from_music(filepath):
    audio, sr = librosa.load(filepath)
    f0 = librosa.yin(audio, fmin=50, fmax=1000)
    f0 = f0[~np.isnan(f0)]
    return f0, sr

def extract_note_sequence(f0, sr, hop_length=512):
    notas = []
    time_per_frame = hop_length / sr

    for i in range(0, len(f0), int(0.5 / time_per_frame)):
        nota = freq_to_note(f0[i])
        if nota:
            notas.append(nota)

    return notas

NOTAS = {
    "C": 261.63, "C#": 277.18, "D": 293.66,
    "D#": 311.13, "E": 329.63, "F": 349.23,
    "F#": 369.99, "G": 392.00, "G#": 415.30,
    "A": 440.00, "A#": 466.16, "B": 493.88
}

NOTAS_NAME = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# --------------------------------------
# BOTÕES NA TELA
# --------------------------------------
btn_mode_music = Button("Escolha a Música", WIDTH//2 - 150, 35, 300, 60, (200, 120, 255), (220, 150, 255))

btn_repeat = Button("Repetir Nota", 80, 120, 220, 60, (255, 160, 80), (255, 190, 120))
btn_next_note = Button("Próxima Nota", 80, 200, 220, 60, (255, 170, 60), (255, 200, 110))
btn_record = Button("Gravar Mic", 80, 280, 220, 60, (255, 100, 100), (255, 150, 150))
btn_tip = Button("Dica", 80, 360, 220, 60, (120, 120, 255), (160, 160, 255))

btn_music1 = Button("Coração de Gelo", WIDTH - 360, 200, 300, 60, (120,180,255), (160,210,255))
btn_music2 = Button("Sem Graça", WIDTH - 360, 280, 300, 60, (120,255,180), (160,255,210))
btn_music3 = Button("Além do que se vê", WIDTH - 360, 360, 300, 60, (255,200,120), (255,220,160))



# --------------------------------------
# VARIÁVEIS DO JOGO
# --------------------------------------
current_note = None
selected_music = None
selecting_music = False
text_note = "Clique em 'Tocar Nota' ou 'Modo Música'."          
text_result = ""
points_penality = "Se acertar só vai pontuar 0.5 pontos por causa da dica."       
points = 0            
music_notes = []
music_challenge_index = 0
music_mode = False
text_tip = ""
used_tip = False

# --------------------------------------
# LOOP PRINCIPAL
# --------------------------------------
running = True
while running:
    screen.fill(DARK_GRAY)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if btn_repeat.clicked(event):
            if current_note:
                play_note(NOTAS[current_note])

        if btn_record.clicked(event):
            if not current_note:
                text_note = "Toque uma nota primeiro!"
            else:
                audio, sr = record_audio()
                detected = detect_pitch(audio, sr)

                if detected is None:
                    text_result  = "Não consegui identificar sua voz."
                else:
                    freq_original = NOTAS[current_note]
                    if abs(detected - freq_original) <= 50:
                        text_result = f"Acertou! Você cantou {detected:.1f} Hz."
                        if used_tip:
                            points += 0.5
                        else:
                            points += 1
                    else:
                        text_result = f"Errou! Sua nota foi {detected:.1f} Hz."

        if btn_mode_music.clicked(event):
            music_mode = False
            selecting_music = True
            text_note = "Escolha a música:"
            text_result = ""

### selecao de musicas
        if selecting_music:
            if btn_music1.clicked(event):
                selected_music = MUSICAS["Coração de Gelo"]
                selecting_music = False
                music_mode = True
                text_note = "Música: Coração de Gelo — Clique em 'Próxima Nota'!"
                f0, sr = extract_notes_from_music(selected_music)
                notas = extract_note_sequence(f0, sr)
                notas = [n for n in notas if n]
                music_notes = notas[:20]
                music_challenge_index = 0
                music_mode = True

            if btn_music2.clicked(event):
                selected_music = MUSICAS["Sem Graça"]
                selecting_music = False
                music_mode = True
                text_note = "Música: Sem Graça — Clique em 'Próxima Nota'!"
                f0, sr = extract_notes_from_music(selected_music)
                notas = extract_note_sequence(f0, sr)
                notas = [n for n in notas if n]
                music_notes = notas[:20]
                music_challenge_index = 0
                music_mode = True

            if btn_music3.clicked(event):
                selected_music = MUSICAS["Além do que se vê"]
                selecting_music = False
                music_mode = True
                text_note = "Música: Além do que se vê — Clique em 'Próxima Nota'!"
                f0, sr = extract_notes_from_music(selected_music)
                notas = extract_note_sequence(f0, sr)
                notas = [n for n in notas if n]
                music_notes = notas[:20]
                music_challenge_index = 0
                music_mode = True

        if btn_next_note.clicked(event) and music_mode:
            if music_challenge_index >= len(music_notes):
                text_note = "Fim das notas da música!"
            else:
                current_note = music_notes[music_challenge_index]
                play_note(NOTAS[current_note])
                text_note  = f"Nota da música: {current_note}. Cante!"
                music_challenge_index += 1
                text_tip = ""
                used_tip = False

        if btn_tip.clicked(event) and music_mode and current_note:
            freq = NOTAS[current_note.replace("#", "")]
            text_tip = f"Dica: {current_note} = {freq:.1f} Hz"
            used_tip = True


    # --------------------------------------
    # DESENHAR BOTÕES
    # --------------------------------------
    if selecting_music:
        btn_music1.draw()
        btn_music2.draw()
        btn_music3.draw()

    btn_repeat.draw()
    btn_record.draw()
    btn_mode_music.draw()
    btn_tip.draw()


    if music_mode:
        btn_next_note.draw()

    # EXIBIR TEXTO
    font = pygame.font.SysFont(None, 30)
    fontDetail = pygame.font.SysFont(None, 20)

    txt1 = font.render(text_note, True, (255, 255, 255))
    screen.blit(txt1, (WIDTH//2 - txt1.get_width()//2, 450))

    txt2 = font.render(text_result, True, (255, 200, 200))
    screen.blit(txt2, (WIDTH//2 - txt2.get_width()//2, 470))

    txt3 = font.render(f"Pontos: {points}", True, (255, 255, 0))
    screen.blit(txt3, (20, 40))

    txt_tip = font.render(text_tip, True, (150, 200, 255))
    screen.blit(txt_tip, (WIDTH//2 - txt_tip.get_width()//2, 490))

    if used_tip:
        txt_penality = fontDetail.render(points_penality, True, (255, 100, 100))
        screen.blit(txt_penality, (20, 20))

    pygame.display.flip()

pygame.quit()