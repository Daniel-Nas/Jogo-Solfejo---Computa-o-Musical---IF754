import pygame
import numpy as np
import threading
import time
import pyaudio
import aubio
import math
import random
from utils import calculate_similarity, is_similar_enough

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

WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Solfejo - Jogo Musical Interativo")

# Cores
BG_DARK = (15, 23, 42)
BG_CARD = (30, 41, 59)
BG_SURFACE = (51, 65, 85)
WHITE = (255, 255, 255)
TEXT_PRIMARY = (248, 250, 252)
TEXT_SECONDARY = (148, 163, 184)

# Cores de Ação
ACCENT = (79, 70, 229)
ACCENT_HOVER = (109, 100, 255)
ACCENT_DARK = (67, 56, 202)

SUCCESS = (16, 185, 129)
SUCCESS_HOVER = (52, 211, 153)
WARNING = (245, 158, 11)
WARNING_HOVER = (251, 191, 36)
DANGER = (220, 38, 38)
DANGER_HOVER = (239, 68, 68)

# Cores Neutras
GRAY_50 = (250, 250, 250)
GRAY_100 = (244, 244, 245)
GRAY_200 = (228, 228, 231)
GRAY_600 = (82, 82, 91)
GRAY_700 = (63, 63, 70)
GRAY_800 = (39, 39, 42)

# Tipografia Aprimorada
FONT_TITLE = pygame.font.SysFont("Arial", 56, bold=True)
FONT_SUBTITLE = pygame.font.SysFont("Arial", 32, bold=True)
FONT_HEADING = pygame.font.SysFont("Arial", 28, bold=True)
FONT = pygame.font.SysFont("Arial", 22)
FONT_SMALL = pygame.font.SysFont("Arial", 18)
FONT_TINY = pygame.font.SysFont("Arial", 14)
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
    def __init__(self, text, rect, color=ACCENT, hover=None, icon=None, font=None):
        self.text = text
        self.rect = pygame.Rect(rect)
        self.color = color
        self.hover = hover or ACCENT_HOVER if color == ACCENT else tuple(min(255, c+20) for c in color)
        self.icon = icon
        self.font = font or FONT
        self.pressed = False

    def draw(self, surf):
        m = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(m)
        is_pressed = is_hover and pygame.mouse.get_pressed()[0]

        # Offset para efeito de press
        offset = 2 if is_pressed else 0
        draw_rect = self.rect.copy()
        draw_rect.y += offset

        # Sombra melhorada (apenas se não pressionado)
        if not is_pressed:
            shadow_rect = draw_rect.copy()
            shadow_rect.y += 5
            shadow_rect.x += 2
            shadow_surf = pygame.Surface((shadow_rect.w, shadow_rect.h), pygame.SRCALPHA)
            pygame.draw.rect(shadow_surf, (0, 0, 0, 60), shadow_surf.get_rect(), border_radius=12)
            surf.blit(shadow_surf, shadow_rect.topleft)

        # Fundo do botão
        col = self.hover if is_hover else self.color
        pygame.draw.rect(surf, col, draw_rect, border_radius=12)

        # Borda sutil para destaque
        border_color = tuple(min(255, c + 20) for c in col)
        pygame.draw.rect(surf, border_color, draw_rect, width=1, border_radius=12)

        if is_hover:
            pygame.draw.rect(surf, tuple(min(255, c+30) for c in col), draw_rect, width=2, border_radius=12)

        # Texto com ícone (se houver)
        text_surf = self.font.render(self.text, True, WHITE)
        if self.icon:
            icon_surf = FONT_HEADING.render(self.icon, True, WHITE)
            total_width = icon_surf.get_width() + 10 + text_surf.get_width()
            start_x = draw_rect.x + (draw_rect.w - total_width) // 2
            surf.blit(icon_surf, (start_x, draw_rect.y + (draw_rect.h - icon_surf.get_height()) // 2))
            surf.blit(text_surf, (start_x + icon_surf.get_width() + 10,
                                 draw_rect.y + (draw_rect.h - text_surf.get_height()) // 2))
        else:
            surf.blit(text_surf, (draw_rect.x + (draw_rect.w - text_surf.get_width()) // 2,
                                 draw_rect.y + (draw_rect.h - text_surf.get_height()) // 2))

    def clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

played_notes = []
currently_playing = False

# Função helper para desenhar cards com sombra
def draw_card(surf, rect, color=BG_CARD, border_radius=16, shadow=True):
    """Desenha um card com sombra e bordas arredondadas"""
    rect = pygame.Rect(rect)

    if shadow:
        # Sombra mais suave e escura
        shadow_rect = rect.copy()
        shadow_rect.y += 4
        shadow_rect.x += 2
        shadow_surf = pygame.Surface((shadow_rect.w, shadow_rect.h), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (0, 0, 0, 80), shadow_surf.get_rect(), border_radius=border_radius)
        surf.blit(shadow_surf, shadow_rect.topleft)

    # Desenha o card
    pygame.draw.rect(surf, color, rect, border_radius=border_radius)

    # Adiciona borda sutil para definição
    border_color = tuple(min(255, c + 15) for c in color)
    pygame.draw.rect(surf, border_color, rect, width=1, border_radius=border_radius)

    return rect

# Função para desenhar texto com sombra
def draw_text_with_shadow(surf, text, font, color, pos, shadow_offset=2):
    """Desenha texto com sombra para melhor legibilidade"""
    shadow = font.render(text, True, (0, 0, 0))
    text_surf = font.render(text, True, color)
    surf.blit(shadow, (pos[0] + shadow_offset, pos[1] + shadow_offset))
    surf.blit(text_surf, pos)

# Função para desenhar badge (vidas, pontos)
def draw_badge(surf, text, rect, color=ACCENT, icon=None):
    """Desenha um badge estilizado"""
    rect = pygame.Rect(rect)

    # Sombra do badge
    shadow_rect = rect.copy()
    shadow_rect.y += 3
    shadow_rect.x += 1
    shadow_surf = pygame.Surface((shadow_rect.w, shadow_rect.h), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, 70), shadow_surf.get_rect(), border_radius=25)
    surf.blit(shadow_surf, shadow_rect.topleft)

    # Fundo do badge
    pygame.draw.rect(surf, color, rect, border_radius=25)

    # Borda para definição
    border_color = tuple(min(255, c + 30) for c in color)
    pygame.draw.rect(surf, border_color, rect, width=1, border_radius=25)

    if icon:
        icon_surf = FONT_HEADING.render(icon, True, WHITE)
        text_surf = FONT_SMALL.render(text, True, WHITE)
        total_width = icon_surf.get_width() + 8 + text_surf.get_width()
        start_x = rect.x + (rect.w - total_width) // 2
        surf.blit(icon_surf, (start_x, rect.y + (rect.h - icon_surf.get_height()) // 2))
        surf.blit(text_surf, (start_x + icon_surf.get_width() + 8,
                             rect.y + (rect.h - text_surf.get_height()) // 2))
    else:
        text_surf = FONT_SMALL.render(text, True, WHITE)
        surf.blit(text_surf, (rect.x + (rect.w - text_surf.get_width()) // 2,
                             rect.y + (rect.h - text_surf.get_height()) // 2))

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

btn_start = Button("INICIAR", (WIDTH//2 - 160, 220, 320, 70), color=ACCENT, font=FONT_HEADING)
btn_rules = Button("REGRAS", (WIDTH//2 - 160, 310, 320, 60), color=BG_SURFACE, font=FONT)
btn_conf = Button("CONFIGURAÇÕES", (WIDTH//2 - 160, 390, 320, 60), color=BG_SURFACE, font=FONT)
btn_back = Button("VOLTAR", (30, HEIGHT-80, 140, 50), color=GRAY_700, font=FONT_SMALL)
btn_menu = Button("MENU", (WIDTH-180, HEIGHT-80, 150, 50), color=DANGER, hover=DANGER_HOVER, font=FONT_SMALL)

btn_repeat = Button("Repetir Notas", (60, 280, 300, 60), color=BG_SURFACE)
btn_action_sing = Button("CANTAR NOTA", (60, 360, 300, 60), color=WARNING)
btn_guess = Button("ADVINHAR MÚSICA", (60, 440, 300, 60), color=ACCENT)

btn_play_target = Button("Ouvir Nota Alvo", (WIDTH-280, 140, 240, 55), color=WARNING)
btn_start_listen = Button("Gravar (Mic)", (WIDTH-280, 215, 240, 55), color=SUCCESS)
btn_skip_confirm = Button("Confirmar", (WIDTH-280, 290, 240, 55), color=ACCENT)


def start_round():
    global current_song_data, current_song_seq, current_index, message, played_notes
    current_song_data = random.choice(BIBLIOTECA)
    current_song_seq = current_song_data.notas 
    current_index = 0
    played_notes = []
    message = "Ouça a primeira nota ou tente advinhar a música."

def draw_menu():
    screen.fill(BG_DARK)

    # Título com efeito de gradiente simulado
    title = "SOLFEJO"
    draw_text_with_shadow(screen, title, FONT_TITLE, ACCENT, (WIDTH//2 - 200, 60), shadow_offset=4)

    # Subtítulo
    subtitle = "Aprenda música de forma divertida"
    subtitle_surf = FONT_SMALL.render(subtitle, True, TEXT_SECONDARY)
    screen.blit(subtitle_surf, (WIDTH//2 - subtitle_surf.get_width()//2, 130))

    # Desenha os botões
    btn_start.draw(screen)
    btn_rules.draw(screen)
    btn_conf.draw(screen)

    # Footer com informações
    footer_text = "Piano Suave (Anti-Clipping) | v2.0"
    footer_surf = FONT_TINY.render(footer_text, True, TEXT_SECONDARY)
    screen.blit(footer_surf, (WIDTH//2 - footer_surf.get_width()//2, HEIGHT - 40))

def draw_rules():
    screen.fill(BG_DARK)

    # Título
    draw_text_with_shadow(screen, "REGRAS DO JOGO", FONT_SUBTITLE, ACCENT, (50, 30), shadow_offset=3)

    # Card com regras
    card_rect = draw_card(screen, (50, 100, WIDTH-100, 450), BG_CARD)

    rules = [
        ("1.", "Você começa com 3 vidas", "Não desperdice tentando adivinhar sem certeza!"),
        ("2.", "Ouça as notas reveladas", "Clique em 'Repetir Notas' para ouvir novamente."),
        ("3.", "Cante a próxima nota", "SEGURE a nota por 1 segundo para desbloquear."),
        ("4.", "Adivinhe a música", "Digite o nome quando achar que sabe qual é!")
    ]

    y = card_rect.y + 30
    for icon, title, desc in rules:
        # Ícone
        icon_surf = FONT_HEADING.render(icon, True, ACCENT)
        screen.blit(icon_surf, (card_rect.x + 30, y))

        # Título
        title_surf = FONT.render(title, True, TEXT_PRIMARY)
        screen.blit(title_surf, (card_rect.x + 80, y + 2))

        # Descrição
        desc_surf = FONT_SMALL.render(desc, True, TEXT_SECONDARY)
        screen.blit(desc_surf, (card_rect.x + 80, y + 32))

        y += 90

    btn_back.draw(screen)

def draw_settings():
    screen.fill(BG_DARK)

    # Título
    draw_text_with_shadow(screen, "CONFIGURAÇÕES", FONT_SUBTITLE, ACCENT, (50, 30), shadow_offset=3)

    # Card de configurações
    card_rect = draw_card(screen, (50, 100, WIDTH-100, 400), BG_CARD)

    # Afinação
    y = card_rect.y + 40
    label_surf = FONT.render("Afinação", True, TEXT_PRIMARY)
    screen.blit(label_surf, (card_rect.x + 40, y))

    value_text = f"+{TUNING_OFFSET} semitons"
    value_surf = FONT_HEADING.render(value_text, True, ACCENT)
    screen.blit(value_surf, (card_rect.x + 40, y + 40))

    desc_surf = FONT_SMALL.render("Ajuste fino da afinação base (A4 = 440Hz)", True, TEXT_SECONDARY)
    screen.blit(desc_surf, (card_rect.x + 40, y + 85))

    # Informações adicionais
    y += 160
    pygame.draw.line(screen, GRAY_700, (card_rect.x + 40, y), (card_rect.right - 40, y), 1)

    y += 30
    info_items = [
        ("Taxa de Amostragem:", f"{SAMPLE_RATE} Hz"),
        ("Duração de Escuta:", f"{LISTEN_DURATION} segundos"),
        ("Estabilidade Requerida:", f"{REQUIRED_STABILITY} segundo")
    ]

    for label, value in info_items:
        label_surf = FONT_SMALL.render(label, True, TEXT_SECONDARY)
        value_surf = FONT_SMALL.render(value, True, TEXT_PRIMARY)
        screen.blit(label_surf, (card_rect.x + 40, y))
        screen.blit(value_surf, (card_rect.x + 350, y))
        y += 35

    btn_back.draw(screen)

def draw_play():
    screen.fill(BG_DARK)

    # Header com badges de status
    header_y = 25
    draw_badge(screen, f"Vidas: {lives}", (50, header_y, 140, 50), DANGER)
    draw_badge(screen, f"Pontos: {score}", (210, header_y, 140, 50), ACCENT)

    # Card principal de informações
    card_main = draw_card(screen, (50, 100, WIDTH-100, 150), BG_CARD)

    # Música (oculta ou revelada)
    nome_show = current_song_data.nome if state == 'gameover' else '???'
    music_label = FONT_SMALL.render("Música:", True, TEXT_SECONDARY)
    music_name = FONT_HEADING.render(nome_show, True, ACCENT)
    screen.blit(music_label, (card_main.x + 30, card_main.y + 25))
    screen.blit(music_name, (card_main.x + 30, card_main.y + 50))

    # Notas liberadas
    displayed = [n[0] for n in current_song_seq[:current_index]]
    txt = " • ".join(displayed) if displayed else "Nenhuma nota revelada ainda"
    notes_label = FONT_SMALL.render("Notas reveladas:", True, TEXT_SECONDARY)
    notes_text = FONT.render(txt, True, TEXT_PRIMARY)
    screen.blit(notes_label, (card_main.x + 30, card_main.y + 95))
    screen.blit(notes_text, (card_main.x + 30, card_main.y + 120))

    # Próxima nota (se houver)
    global play_here_button
    if current_index < len(current_song_seq):
        card_next = draw_card(screen, (400, 100, 550, 150), BG_SURFACE)
        next_label = FONT_SMALL.render("Próxima Nota:", True, TEXT_SECONDARY)
        next_value = FONT_TITLE.render("?", True, WARNING)
        screen.blit(next_label, (card_next.x + 30, card_next.y + 25))
        screen.blit(next_value, (card_next.x + 30, card_next.y + 50))

        btn_play_here = Button("Ouvir", (card_next.x + 200, card_next.y + 60, 150, 50), color=WARNING, font=FONT_SMALL)
        btn_play_here.draw(screen)
        play_here_button = btn_play_here

    # Botões de ação
    btn_repeat.draw(screen)
    btn_action_sing.draw(screen)
    btn_guess.draw(screen)

    # Campo de input melhorado
    input_y = 540
    input_label = FONT_SMALL.render("Adivinhe a música:", True, TEXT_SECONDARY)
    screen.blit(input_label, (400, input_y - 25))

    input_rect = draw_card(screen, (400, input_y, 540, 55), BG_SURFACE, border_radius=12)
    if input_active:
        pygame.draw.rect(screen, ACCENT, input_rect, width=2, border_radius=12)

    display_text = user_text if (input_active or user_text) else "Digite aqui..."
    col = TEXT_PRIMARY if input_active else TEXT_SECONDARY
    text_surf = FONT.render(display_text, True, col)
    screen.blit(text_surf, (input_rect.x + 20, input_rect.y + 15))

    # Mensagem de feedback
    if message:
        msg_color = WARNING if "tempo" in message.lower() else SUCCESS if "acertou" in message.lower() else TEXT_PRIMARY
        msg_surf = FONT_SMALL.render(message, True, msg_color)
        screen.blit(msg_surf, (50, HEIGHT - 50))

    # Botão para retornar ao menu
    btn_menu.draw(screen)

def draw_detector():
    screen.fill(BG_DARK)

    # Título
    draw_text_with_shadow(screen, "DETECTOR DE PITCH", FONT_SUBTITLE, ACCENT, (50, 30), shadow_offset=3)

    # Card principal - Nota alvo
    target = current_song_seq[current_index][0] if current_index < len(current_song_seq) else "-"
    card_target = draw_card(screen, (50, 100, WIDTH-350, 200), BG_CARD)

    target_label = FONT_SMALL.render("Cante e SEGURE esta nota:", True, TEXT_SECONDARY)
    screen.blit(target_label, (card_target.x + 30, card_target.y + 25))

    target_surf = FONT_TITLE.render(target, True, WARNING)
    screen.blit(target_surf, (card_target.x + 30, card_target.y + 60))

    instruction = FONT_TINY.render("Mantenha a nota estável por 1 segundo", True, TEXT_SECONDARY)
    screen.blit(instruction, (card_target.x + 30, card_target.y + 160))

    # Card de detecção
    card_detect = draw_card(screen, (50, 320, WIDTH-350, 250), BG_SURFACE)

    detect_label = FONT_SMALL.render("Detecção em tempo real:", True, TEXT_SECONDARY)
    screen.blit(detect_label, (card_detect.x + 30, card_detect.y + 25))

    if detected_name:
        # Nota detectada
        detected_surf = FONT_HEADING.render(detected_name, True, SUCCESS)
        freq_surf = FONT_SMALL.render(f"{detected_freq:.1f} Hz", True, TEXT_SECONDARY)
        screen.blit(detected_surf, (card_detect.x + 30, card_detect.y + 60))
        screen.blit(freq_surf, (card_detect.x + 30, card_detect.y + 100))

        # Indicador visual de correspondência
        note_only = ''.join([c for c in detected_name if not c.isdigit()])
        if note_only == target:
            match_indicator = FONT.render("Nota Correta!", True, SUCCESS)
            pygame.draw.rect(screen, SUCCESS, (card_detect.x + 30, card_detect.y + 140, 300, 8), border_radius=4)
        else:
            match_indicator = FONT.render("Continue tentando...", True, WARNING)
            pygame.draw.rect(screen, WARNING, (card_detect.x + 30, card_detect.y + 140, 300, 8), border_radius=4)
        screen.blit(match_indicator, (card_detect.x + 30, card_detect.y + 165))
    else:
        no_detect = FONT.render("Aguardando entrada...", True, TEXT_SECONDARY)
        screen.blit(no_detect, (card_detect.x + 30, card_detect.y + 60))

    # Mensagem de status
    msg_y = card_detect.y + 210
    msg_color = WARNING if detector.running else TEXT_SECONDARY
    if detector_result is True: msg_color = SUCCESS
    elif detector_result is False: msg_color = DANGER
    msg_surf = FONT_SMALL.render(message, True, msg_color)
    screen.blit(msg_surf, (card_detect.x + 30, msg_y))

    # Botões de controle (lado direito)
    btn_play_target.draw(screen)
    btn_start_listen.draw(screen)
    btn_skip_confirm.draw(screen)

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
            if btn_menu.clicked(event):
                detector.stop()
                input_active = False
                user_text = ""
                message = ""
                state = 'menu'

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
                    guess = user_text.strip()
                    real = current_song_data.nome or ""

                    if is_similar_enough(guess, real):
                        score += 5
                        similarity = calculate_similarity(guess, real)

                        # Mensagem diferente se acertou exatamente ou com pequenos erros
                        if similarity == 1.0:
                            message = f"✓ PERFEITO: {current_song_data.nome}!"
                        else:
                            message = f"✓ ACERTOU: {current_song_data.nome}!"

                        start_round()
                        if current_song_seq:
                            n = current_song_seq[0]
                            threading.Thread(target=play_note, args=(NOTE_FREQS[n[0]], n[1]), daemon=True).start()
                    else:
                        lives -= 1
                        similarity = calculate_similarity(guess, real)
                        message = f"✗ Errou! Era '{current_song_data.nome}'. Vidas: {lives}"
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
        screen.fill(BG_DARK)

        # Card central de game over
        card = draw_card(screen, (WIDTH//2 - 300, HEIGHT//2 - 200, 600, 400), BG_CARD)

        # Título
        title_surf = FONT_TITLE.render("FIM DE JOGO", True, DANGER)
        screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, card.y + 50))

        # Pontuação com destaque
        score_label = FONT_SMALL.render("Pontuação Final", True, TEXT_SECONDARY)
        screen.blit(score_label, (WIDTH//2 - score_label.get_width()//2, card.y + 140))

        score_surf = FONT_TITLE.render(f"{score}", True, ACCENT)
        screen.blit(score_surf, (WIDTH//2 - score_surf.get_width()//2, card.y + 170))

        # Música revelada
        music_name = current_song_data.nome if current_song_data else "Desconhecida"
        music_label = FONT_SMALL.render("A música era:", True, TEXT_SECONDARY)
        music_surf = FONT_HEADING.render(music_name, True, WARNING)
        screen.blit(music_label, (WIDTH//2 - music_label.get_width()//2, card.y + 250))
        screen.blit(music_surf, (WIDTH//2 - music_surf.get_width()//2, card.y + 280))

        btn_back.draw(screen)

    pygame.display.flip()
    CLOCK.tick(30)

pygame.quit()