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
button_cooldown_until = 0

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
        self.BUFFER_SIZE = 8192
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

# Cores - Tema Gradiente Roxo-Azul
BG_DARK = (20, 15, 35)  # Roxo escuro base
BG_CARD = (45, 35, 70)  # Roxo médio
BG_SURFACE = (60, 50, 90)  # Roxo claro
WHITE = (255, 255, 255)
TEXT_PRIMARY = (248, 250, 252)
TEXT_SECONDARY = (180, 170, 220)

# Cores de Ação - Gradiente Roxo-Azul
ACCENT = (120, 80, 200)  # Roxo vibrante
ACCENT_HOVER = (140, 100, 240)  # Roxo claro
ACCENT_DARK = (100, 60, 180)  # Roxo escuro

SUCCESS = (80, 200, 180)  # Azul-verde
SUCCESS_HOVER = (100, 220, 200)
WARNING = (255, 180, 100)  # Laranja dourado
WARNING_HOVER = (255, 200, 120)
DANGER = (240, 80, 120)  # Rosa-vermelho
DANGER_HOVER = (255, 100, 140)

# Cores Neutras - Ajustadas para tema roxo-azul
GRAY_50 = (250, 250, 250)
GRAY_100 = (244, 244, 245)
GRAY_200 = (228, 228, 231)
GRAY_600 = (100, 90, 130)  # Roxo acinzentado
GRAY_700 = (80, 70, 110)   # Roxo escuro
GRAY_800 = (50, 40, 70)    # Roxo muito escuro

# Tipografia Aprimorada - Montserrat
def get_font(name, size, bold=False):
    """Tenta carregar uma fonte, com fallback para alternativas"""
    import os
    
    # Primeiro, tenta carregar de arquivo local (se existir)
    font_files = [
        f"fonts/{name}-Bold.ttf" if bold else f"fonts/{name}-Regular.ttf",
        f"fonts/{name}Bold.ttf" if bold else f"fonts/{name}Regular.ttf",
        f"fonts/{name}.ttf",
    ]
    
    for font_file in font_files:
        if os.path.exists(font_file):
            try:
                return pygame.font.Font(font_file, size)
            except:
                pass
    
    # Se não encontrou arquivo, tenta fontes do sistema
    # Montserrat pode ter variações no nome dependendo do sistema
    if name == "Montserrat":
        font_names = [
            "Montserrat", 
            "Montserrat Bold" if bold else "Montserrat Regular",
            "Montserrat-Bold" if bold else "Montserrat-Regular",
            "Segoe UI",  # Fonte moderna similar
            "Calibri",   # Fonte moderna similar
            "Arial"      # Fallback final
        ]
    else:
        font_names = [name, "Segoe UI", "Calibri", "Arial"]
    
    for font_name in font_names:
        try:
            if bold:
                font = pygame.font.SysFont(font_name, size, bold=True)
            else:
                font = pygame.font.SysFont(font_name, size)
            # Testa se a fonte foi carregada corretamente
            test_surf = font.render("Test", True, (255, 255, 255))
            if test_surf:
                return font
        except:
            continue
    
    # Fallback final: fonte padrão do pygame
    return pygame.font.Font(None, size)

# Inicializa as fontes com Montserrat (ou fallback)
FONT_TITLE = get_font("Montserrat", 72, bold=True)  # Aumentado de 56 para 72
FONT_TITLE_LARGE = get_font("Montserrat", 96, bold=True)  # Fonte extra grande para animação
FONT_SUBTITLE = get_font("Montserrat", 32, bold=True)
FONT_HEADING = get_font("Montserrat", 28, bold=True)
FONT = get_font("Montserrat", 22, bold=False)
FONT_SMALL = get_font("Montserrat", 18, bold=False)
FONT_TINY = get_font("Montserrat", 14, bold=False)
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
        # Hover mais sutil - apenas um leve brilho
        self.hover = hover or tuple(min(255, c + 15) for c in color) if hover is None else hover
        self.icon = icon
        self.font = font or FONT
        self.pressed = False

    def draw(self, surf):
        m = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(m)
        is_pressed = is_hover and pygame.mouse.get_pressed()[0]

        # Offset muito sutil para efeito de press
        offset = 1 if is_pressed else 0
        draw_rect = self.rect.copy()
        draw_rect.y += offset

        # Border radius mais arredondado (formato de pílula - usa metade da altura)
        # Mas limita a um máximo para não ficar muito extremo
        border_radius = min(draw_rect.h // 2, 35)

        # Sombra muito sutil e suave (minimalista)
        if not is_pressed:
            shadow_rect = draw_rect.copy()
            shadow_rect.y += 2
            shadow_surf = pygame.Surface((shadow_rect.w, shadow_rect.h), pygame.SRCALPHA)
            # Sombra muito mais suave e translúcida
            pygame.draw.rect(shadow_surf, (0, 0, 0, 20), shadow_surf.get_rect(), border_radius=border_radius)
            surf.blit(shadow_surf, shadow_rect.topleft)

        # Cor base com leve ajuste no hover
        base_col = self.hover if is_hover else self.color
        
        # Gradiente muito mais sutil (quase imperceptível)
        col_start = tuple(min(255, c + 8) for c in base_col)
        col_end = tuple(max(0, c - 5) for c in base_col)
        
        # Desenha o botão arredondado com gradiente
        # Cria superfície temporária para aplicar o gradiente com border radius
        button_surf = pygame.Surface((draw_rect.w, draw_rect.h), pygame.SRCALPHA)
        
        # Desenha o gradiente
        draw_gradient(button_surf, (0, 0, draw_rect.w, draw_rect.h), col_start, col_end, vertical=True)
        
        # Cria máscara arredondada
        mask = pygame.Surface((draw_rect.w, draw_rect.h), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, draw_rect.w, draw_rect.h), border_radius=border_radius)
        
        # Aplica máscara ao botão (multiplica alphas para criar bordas arredondadas)
        button_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        # Aplica leve transparência se não estiver em hover
        if not is_hover:
            button_surf.set_alpha(240)
        
        surf.blit(button_surf, draw_rect.topleft)

        # Sem bordas - design minimalista
        # Apenas uma borda muito sutil no hover
        if is_hover and not is_pressed:
            border_color = tuple(min(255, c + 20) for c in base_col)
            pygame.draw.rect(surf, border_color, draw_rect, width=1, border_radius=border_radius)

        # Texto com ícone (se houver) - cor mais suave
        text_color = WHITE if is_hover else (245, 245, 250)  # Branco levemente acinzentado quando não hover
        text_surf = self.font.render(self.text, True, text_color)
        
        if self.icon:
            icon_surf = FONT_HEADING.render(self.icon, True, text_color)
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
played_past_notes = []
currently_playing = False

# Função para desenhar gradiente
def draw_gradient(surf, rect, color_start, color_end, vertical=True):
    """Desenha um gradiente linear no retângulo especificado"""
    rect = pygame.Rect(rect)
    gradient_surf = pygame.Surface((rect.w, rect.h))
    
    if vertical:
        for y in range(rect.h):
            ratio = y / rect.h
            r = int(color_start[0] * (1 - ratio) + color_end[0] * ratio)
            g = int(color_start[1] * (1 - ratio) + color_end[1] * ratio)
            b = int(color_start[2] * (1 - ratio) + color_end[2] * ratio)
            pygame.draw.line(gradient_surf, (r, g, b), (0, y), (rect.w, y))
    else:
        for x in range(rect.w):
            ratio = x / rect.w
            r = int(color_start[0] * (1 - ratio) + color_end[0] * ratio)
            g = int(color_start[1] * (1 - ratio) + color_end[1] * ratio)
            b = int(color_start[2] * (1 - ratio) + color_end[2] * ratio)
            pygame.draw.line(gradient_surf, (r, g, b), (x, 0), (x, rect.h))
    
    surf.blit(gradient_surf, rect.topleft)

# Função helper para desenhar cards com sombra e gradiente (minimalista)
def draw_card(surf, rect, color=BG_CARD, border_radius=20, shadow=True, gradient=False):
    """Desenha um card minimalista com sombra suave e bordas arredondadas"""
    rect = pygame.Rect(rect)

    if shadow:
        # Sombra muito mais suave e minimalista
        shadow_rect = rect.copy()
        shadow_rect.y += 2
        shadow_surf = pygame.Surface((shadow_rect.w, shadow_rect.h), pygame.SRCALPHA)
        # Sombra muito translúcida
        pygame.draw.rect(shadow_surf, (0, 0, 0, 15), shadow_surf.get_rect(), border_radius=border_radius)
        surf.blit(shadow_surf, shadow_rect.topleft)

    # Desenha o card com ou sem gradiente (muito mais sutil)
    if gradient:
        # Gradiente muito mais sutil
        color_start = tuple(min(255, c + 10) for c in color)
        color_end = tuple(max(0, c - 8) for c in color)
        draw_gradient(surf, rect, color_start, color_end, vertical=True)
    else:
        pygame.draw.rect(surf, color, rect, border_radius=border_radius)

    # Sem borda ou borda muito sutil (minimalista)
    # Removido para design mais limpo

    return rect

# Função para desenhar texto com sombra
def draw_text_with_shadow(surf, text, font, color, pos, shadow_offset=2):
    """Desenha texto com sombra para melhor legibilidade"""
    shadow = font.render(text, True, (0, 0, 0))
    text_surf = font.render(text, True, color)
    surf.blit(shadow, (pos[0] + shadow_offset, pos[1] + shadow_offset))
    surf.blit(text_surf, pos)

# Função para desenhar badge (vidas, pontos) com gradiente minimalista
def draw_badge(surf, text, rect, color=ACCENT, icon=None):
    """Desenha um badge minimalista com gradiente suave"""
    rect = pygame.Rect(rect)

    # Sombra muito suave
    shadow_rect = rect.copy()
    shadow_rect.y += 1
    shadow_surf = pygame.Surface((shadow_rect.w, shadow_rect.h), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, 12), shadow_surf.get_rect(), border_radius=20)
    surf.blit(shadow_surf, shadow_rect.topleft)

    # Fundo do badge com gradiente muito sutil
    color_start = tuple(min(255, c + 8) for c in color)
    color_end = tuple(max(0, c - 5) for c in color)
    draw_gradient(surf, rect, color_start, color_end, vertical=True)

    # Sem borda - design minimalista

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


def cents_difference(freq, target_freq):
    """Diferença em cents entre a frequência atual e a frequência alvo"""
    if freq <= 0 or target_freq <= 0:
        return None
    return 1200 * math.log2(freq / target_freq)


def draw_needle_gauge(surf, rect, current_freq, target_freq, tolerance_hz=None, min_freq=0.0, max_freq=500.0):
    """Mostra uma agulha absoluta de 0 Hz a 400 Hz, destacando a posição do alvo."""
    rect = pygame.Rect(rect)
    pygame.draw.rect(surf, BG_CARD, rect, border_radius=16)
    pygame.draw.rect(surf, GRAY_700, rect, width=1, border_radius=16)

    # Normaliza limites
    min_f = max(0.0, float(min_freq))
    max_f = max(min_f + 1.0, float(max_freq))  # evita divisão por zero
    span = max_f - min_f

    # Valores atuais
    curr = None if current_freq is None or current_freq <= 0 else float(current_freq)
    tgt = None if target_freq is None or target_freq <= 0 else float(target_freq)

    tol = tolerance_hz if tolerance_hz is not None else 5.0

    bar_x = rect.x + 30
    bar_w = rect.w - 60
    bar_y = rect.y + rect.h // 2 - 6
    bar_h = 12

    pygame.draw.rect(surf, GRAY_800, (bar_x, bar_y, bar_w, bar_h), border_radius=8)

    # Faixa verde em torno do alvo
    if tgt is not None:
        tgt_clamped = max(min_f, min(max_f, tgt))
        tol_left = max(min_f, tgt_clamped - tol)
        tol_right = min(max_f, tgt_clamped + tol)
        tol_px_start = bar_x + int((tol_left - min_f) / span * bar_w)
        tol_px_end = bar_x + int((tol_right - min_f) / span * bar_w)
        pygame.draw.rect(surf, SUCCESS, (tol_px_start, bar_y, tol_px_end - tol_px_start, bar_h), border_radius=8)

    # Agulha (frequência atual)
    if curr is not None:
        curr_clamped = max(min_f, min(max_f, curr))
        needle_x = bar_x + int((curr_clamped - min_f) / span * bar_w)
        pygame.draw.line(surf, WARNING, (needle_x, bar_y - 14), (needle_x, bar_y + bar_h + 14), 3)
        pygame.draw.circle(surf, WHITE, (needle_x, bar_y + bar_h + 16), 6)

    # Marcador do alvo
    if tgt is not None:
        target_x = bar_x + int((tgt_clamped - min_f) / span * bar_w)
        pygame.draw.line(surf, ACCENT, (target_x, bar_y - 10), (target_x, bar_y + bar_h + 10), 2)
        pygame.draw.circle(surf, ACCENT, (target_x, bar_y - 12), 4)

    label_font = FONT_TINY
    left = label_font.render(f"{min_f:.0f} Hz", True, TEXT_SECONDARY)
    mid_val = (min_f + max_f) / 2
    mid = label_font.render(f"{mid_val:.0f} Hz", True, TEXT_SECONDARY)
    right = label_font.render(f"{max_f:.0f} Hz", True, TEXT_SECONDARY)
    surf.blit(left, (bar_x, bar_y + 20))
    surf.blit(mid, (bar_x + (bar_w - mid.get_width()) // 2, bar_y + 20))
    surf.blit(right, (bar_x + bar_w - right.get_width(), bar_y + 20))

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
    global message, detected_name, detected_freq, detector_result, detected_deviation_hz

    detected_name = None
    detected_freq = None
    detector_result = None
    detected_deviation_hz = None

    while currently_playing:
        time.sleep(0.01)

    detector.start()
    message = "Prepare-se... Cante e SEGURE a nota!"
    target_freq = NOTE_FREQS.get(target_note_name)
    tolerance_hz = 30.0
    
    session_start_time = time.time()
    stable_start_time = None 
    found_match = False
    
    while time.time() - session_start_time < LISTEN_DURATION:
        note_completa = detector.get_note()
        freq = detector.get_freq()
        
        if note_completa:
            detected_name = note_completa
            detected_freq = freq
            detected_deviation_hz = (freq - target_freq) if target_freq else None
            note_only_name = ''.join([c for c in note_completa if not c.isdigit()])
            within_tolerance = (target_freq is not None and
                                 tolerance_hz is not None and
                                 detected_deviation_hz is not None and
                                 abs(detected_deviation_hz) <= tolerance_hz)

            if within_tolerance:
                if stable_start_time is None:
                    stable_start_time = time.time()

                elapsed = time.time() - stable_start_time
                message = f"Mantenha por {REQUIRED_STABILITY - elapsed:.1f}s" if elapsed < REQUIRED_STABILITY else "Nota estável!"

                if elapsed >= REQUIRED_STABILITY:
                    detector_result = True
                    message = f"Nota {target_note_name} confirmada."
                    found_match = True
                    break
            else:
                stable_start_time = None
                if detected_deviation_hz is None:
                    message = f"Detectado: {detected_name}. Alvo: {target_note_name}"
                else:
                    direction = "Suba" if detected_deviation_hz < 0 else "Desça"
                    message = f"{direction} {abs(detected_deviation_hz):.2f} Hz até {target_note_name}"
        else:
            stable_start_time = None
            detected_deviation_hz = None
            if target_freq:
                message = f"Silêncio... alvo {target_note_name} ({target_freq:.1f} Hz)"
            else:
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
guess_modal_open = False
currently_playing = False
detected_name = None
detected_freq = None
detected_deviation_hz = None
detector_result = None
show_success_animation = False
success_animation_start_time = 0
SUCCESS_ANIMATION_DURATION = 2.0  # Duração em segundos

btn_start = Button("INICIAR", (WIDTH//2 - 160, 220, 320, 70), color=ACCENT, font=FONT_HEADING)
btn_rules = Button("REGRAS", (WIDTH//2 - 160, 310, 320, 60), color=(100, 70, 150), hover=(120, 90, 170), font=FONT)
btn_conf = Button("CONFIGURAÇÕES", (WIDTH//2 - 160, 390, 320, 60), color=(100, 70, 150), hover=(120, 90, 170), font=FONT)
btn_back = Button("VOLTAR", (30, HEIGHT-80, 140, 50), color=GRAY_700, hover=(100, 90, 130), font=FONT_SMALL)
btn_menu = Button("MENU", (WIDTH-180, HEIGHT-80, 150, 50), color=DANGER, hover=DANGER_HOVER, font=FONT_SMALL)

btn_repeat = Button("Repetir Notas", (60, 280, 300, 60), color=(80, 60, 120), hover=(100, 80, 140))
btn_action_sing = Button("CANTAR NOTA", (60, 360, 300, 60), color=WARNING)
btn_guess = Button("ADVINHAR MÚSICA", (60, 440, 300, 60), color=ACCENT)

btn_play_target = Button("Ouvir Nota Alvo", (WIDTH-280, 140, 240, 55), color=WARNING)
btn_start_listen = Button("Gravar (Mic)", (WIDTH-280, 215, 240, 55), color=SUCCESS)
btn_skip_confirm = Button("Confirmar", (WIDTH-280, 290, 240, 55), color=ACCENT)

# Botões do modal de adivinhar música (serão criados dinamicamente)
btn_modal_confirm = None
btn_modal_cancel = None


def start_round():
    global current_song_data, current_song_seq, current_index, message, played_notes, played_past_notes
    current_song_data = random.choice(BIBLIOTECA)
    current_song_seq = current_song_data.notas 
    current_index = 0
    played_notes = []
    played_past_notes = []
    message = "Ouça a primeira nota ou tente advinhar a música."

def draw_note_symbol(surf, x, y, size=30, color=(255, 255, 255)):
    """Desenha uma nota musical decorativa"""
    # Cria uma superfície com alpha para transparência
    note_surf = pygame.Surface((size * 2, size * 3), pygame.SRCALPHA)
    # Cabeça da nota (círculo)
    pygame.draw.circle(note_surf, (*color, 150), (size, size // 2), size // 2, 2)
    # Haste da nota
    pygame.draw.line(note_surf, (*color, 150), (size, size), (size, size * 2), 2)
    surf.blit(note_surf, (x - size, y - size // 2))

def draw_musical_staff(surf, x, y, width, height):
    """Desenha uma pauta musical decorativa"""
    staff_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    line_color = (255, 255, 255, 30)
    line_spacing = height // 5
    for i in range(5):
        pygame.draw.line(staff_surf, line_color, (0, i * line_spacing), 
                        (width, i * line_spacing), 1)
    surf.blit(staff_surf, (x, y))

def draw_menu():
    # Background com gradiente roxo-azul
    purple_start = (60, 20, 80)  # Roxo escuro
    blue_end = (20, 40, 100)     # Azul escuro
    draw_gradient(screen, (0, 0, WIDTH, HEIGHT), purple_start, blue_end, vertical=True)

    # Elementos musicais decorativos no fundo
    # Pautas musicais sutis
    draw_musical_staff(screen, 50, 50, 200, 80)
    draw_musical_staff(screen, WIDTH - 250, 50, 200, 80)
    draw_musical_staff(screen, 50, HEIGHT - 150, 200, 80)
    draw_musical_staff(screen, WIDTH - 250, HEIGHT - 150, 200, 80)

    # Notas musicais decorativas
    note_color = (255, 255, 255)
    draw_note_symbol(screen, 120, 100, 25, note_color)
    draw_note_symbol(screen, 180, 120, 25, note_color)
    draw_note_symbol(screen, WIDTH - 180, 100, 25, note_color)
    draw_note_symbol(screen, WIDTH - 120, 120, 25, note_color)
    draw_note_symbol(screen, 120, HEIGHT - 100, 25, note_color)
    draw_note_symbol(screen, 180, HEIGHT - 80, 25, note_color)
    draw_note_symbol(screen, WIDTH - 180, HEIGHT - 100, 25, note_color)
    draw_note_symbol(screen, WIDTH - 120, HEIGHT - 80, 25, note_color)

    # Título principal com design musical melhorado e animação
    title = "SOLFEJO"
    title_x = WIDTH // 2
    title_y = 50
    
    # Animação de pulsação baseada no tempo
    current_time = pygame.time.get_ticks()
    # Pulsação suave (ciclo de 2 segundos)
    pulse_speed = 0.002
    pulse = 1.0 + math.sin(current_time * pulse_speed) * 0.08  # Varia entre 0.92 e 1.08
    # Brilho pulsante
    glow_intensity = 0.5 + math.sin(current_time * pulse_speed) * 0.5  # Varia entre 0 e 1
    
    # Calcula o tamanho animado
    base_size = 96
    animated_size = int(base_size * pulse)
    animated_font = get_font("Montserrat", animated_size, bold=True)
    
    # Renderiza o título com tamanho animado
    title_surf = animated_font.render(title, True, ACCENT)
    title_width = title_surf.get_width()
    title_height = title_surf.get_height()
    
    # Desenha o título com múltiplas camadas para efeito de brilho animado
    # Sombra principal (múltiplas camadas para suavidade)
    for i in range(3):
        shadow_offset = 4 - i
        shadow_alpha = int(40 - i * 12)
        shadow_surf = animated_font.render(title, True, (0, 0, 0))
        shadow_surf.set_alpha(shadow_alpha)
        screen.blit(shadow_surf, (title_x - title_width//2 + shadow_offset, title_y + shadow_offset))
    
    # Efeito de brilho pulsante (múltiplas camadas)
    num_glow_layers = 3
    for i in range(num_glow_layers):
        glow_alpha = int(80 * glow_intensity / (i + 1))
        glow_offset = i * 2
        glow_color = tuple(min(255, c + int(30 * (i + 1))) for c in ACCENT)
        glow_surf = animated_font.render(title, True, glow_color)
        glow_surf.set_alpha(glow_alpha)
        screen.blit(glow_surf, (title_x - title_width//2 - glow_offset, title_y - glow_offset))
    
    # Camada base do título
    screen.blit(title_surf, (title_x - title_width//2, title_y))
    
    # Camada de brilho superior (mais clara) com animação
    glow_color = tuple(min(255, c + int(40 * glow_intensity)) for c in ACCENT)
    glow_surf = animated_font.render(title, True, glow_color)
    glow_surf.set_alpha(int(150 * glow_intensity))
    screen.blit(glow_surf, (title_x - title_width//2 - 2, title_y - 2))

    # Ícones musicais ao lado do título com animação suave
    music_icon = "♪"
    icon_size = int(56 * pulse)
    icon_font = get_font("Montserrat", icon_size, bold=True)
    icon_surf = icon_font.render(music_icon, True, (200, 180, 255))
    icon_alpha = int(200 + 55 * glow_intensity)
    icon_surf.set_alpha(icon_alpha)
    
    # Ícone esquerdo (com leve rotação/posição animada)
    icon_offset_y = int(math.sin(current_time * pulse_speed * 1.5) * 3)
    screen.blit(icon_surf, (title_x - title_width//2 - 90, title_y + 10 + icon_offset_y))
    # Ícone direito
    icon_offset_y2 = int(math.sin(current_time * pulse_speed * 1.5 + math.pi) * 3)
    screen.blit(icon_surf, (title_x + title_width//2 + 30, title_y + 10 + icon_offset_y2))
    
    # Adiciona mais ícones musicais menores decorativos com animação
    small_icon = "♫"
    small_icon_size = int(32 * (1.0 + math.sin(current_time * pulse_speed * 2) * 0.1))
    small_icon_font = get_font("Montserrat", small_icon_size, bold=True)
    small_icon_surf = small_icon_font.render(small_icon, True, (180, 160, 240))
    small_icon_surf.set_alpha(int(180 + 75 * glow_intensity))
    screen.blit(small_icon_surf, (title_x - title_width//2 - 120, title_y + 20))
    screen.blit(small_icon_surf, (title_x + title_width//2 + 60, title_y + 20))

    # Subtítulo melhorado com estilo musical (ajustado para o título maior)
    subtitle = "Aprenda música de forma divertida"
    subtitle_y = title_y + title_height + 20  # Posiciona abaixo do título animado
    subtitle_surf = FONT_SMALL.render(subtitle, True, (220, 200, 255))
    screen.blit(subtitle_surf, (WIDTH//2 - subtitle_surf.get_width()//2, subtitle_y))
    
    # Linha decorativa sob o subtítulo
    line_y = subtitle_y + 25
    line_surf = pygame.Surface((300, 2), pygame.SRCALPHA)
    line_surf.fill((180, 160, 220, 100))
    screen.blit(line_surf, (WIDTH//2 - 150, line_y))

    # Desenha os botões
    btn_start.draw(screen)
    btn_rules.draw(screen)
    btn_conf.draw(screen)

    # Footer melhorado com ícone musical
    footer_text = "♪ Piano Suave (Anti-Clipping) | v2.0 ♪"
    footer_surf = FONT_TINY.render(footer_text, True, (180, 170, 220))
    screen.blit(footer_surf, (WIDTH//2 - footer_surf.get_width()//2, HEIGHT - 40))

def draw_rules():
    # Background com gradiente roxo-azul
    purple_start = (60, 20, 80)  # Roxo escuro
    blue_end = (20, 40, 100)     # Azul escuro
    draw_gradient(screen, (0, 0, WIDTH, HEIGHT), purple_start, blue_end, vertical=True)

    # Título
    draw_text_with_shadow(screen, "REGRAS DO JOGO", FONT_SUBTITLE, ACCENT, (50, 30), shadow_offset=3)

    # Card com regras e gradiente
    card_rect = draw_card(screen, (50, 100, WIDTH-100, 450), BG_CARD, gradient=True)

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
    # Background com gradiente roxo-azul
    purple_start = (60, 20, 80)  # Roxo escuro
    blue_end = (20, 40, 100)     # Azul escuro
    draw_gradient(screen, (0, 0, WIDTH, HEIGHT), purple_start, blue_end, vertical=True)

    # Título
    draw_text_with_shadow(screen, "CONFIGURAÇÕES", FONT_SUBTITLE, ACCENT, (50, 30), shadow_offset=3)

    # Card de configurações com gradiente
    card_rect = draw_card(screen, (50, 100, WIDTH-100, 400), BG_CARD, gradient=True)

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

def draw_success_animation():
    """Desenha uma animação visual quando o jogador acerta uma nota"""
    global show_success_animation, success_animation_start_time
    
    if not show_success_animation:
        return
    
    # Calcula o tempo decorrido desde o início da animação (em milissegundos)
    current_time = pygame.time.get_ticks()
    elapsed_ms = current_time - success_animation_start_time
    elapsed = elapsed_ms / 1000.0  # Converte para segundos
    
    # Se passou o tempo, desativa a animação
    if elapsed >= SUCCESS_ANIMATION_DURATION:
        show_success_animation = False
        return
    
    # Calcula a opacidade (fade out nos últimos 0.5 segundos)
    fade_start = SUCCESS_ANIMATION_DURATION - 0.5
    if elapsed > fade_start:
        alpha = int(255 * (1 - (elapsed - fade_start) / 0.5))
    else:
        alpha = 255
    
    # Calcula o tamanho pulsante da animação
    pulse_speed = 0.02
    pulse = 1.0 + math.sin(pygame.time.get_ticks() * pulse_speed) * 0.2
    
    # Overlay semi-transparente verde
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay_alpha = int(30 * (alpha / 255))
    overlay.fill((0, 255, 100, overlay_alpha))
    screen.blit(overlay, (0, 0))
    
    # Card central com mensagem de sucesso
    card_width = 500
    card_height = 250
    card_x = (WIDTH - card_width) // 2
    card_y = (HEIGHT - card_height) // 2
    
    # Efeito de escala pulsante
    scale = pulse
    scaled_width = int(card_width * scale)
    scaled_height = int(card_height * scale)
    scaled_x = (WIDTH - scaled_width) // 2
    scaled_y = (HEIGHT - scaled_height) // 2
    
    # Card com gradiente verde
    card_surf = pygame.Surface((scaled_width, scaled_height), pygame.SRCALPHA)
    card_rect = pygame.Rect(0, 0, scaled_width, scaled_height)
    
    # Gradiente verde brilhante
    color_start = (50, 255, 150)
    color_end = (80, 220, 120)
    draw_gradient(card_surf, card_rect, color_start, color_end, vertical=True)
    
    # Borda brilhante
    pygame.draw.rect(card_surf, (100, 255, 180, alpha), card_rect, width=4, border_radius=25)
    
    card_surf.set_alpha(alpha)
    screen.blit(card_surf, (scaled_x, scaled_y))
    
    # Ícone de check/certo grande
    check_size = int(80 * scale)
    check_font = get_font("Montserrat", check_size, bold=True)
    check_text = "✓"
    check_surf = check_font.render(check_text, True, (255, 255, 255))
    check_alpha = alpha
    check_surf.set_alpha(check_alpha)
    screen.blit(check_surf, (WIDTH//2 - check_surf.get_width()//2, scaled_y + 40))
    
    # Texto "NOTA ACERTADA!"
    success_text = "NOTA ACERTADA!"
    success_font = FONT_TITLE
    success_surf = success_font.render(success_text, True, (255, 255, 255))
    success_shadow = success_font.render(success_text, True, (0, 0, 0))
    
    # Aplica alpha
    success_surf_temp = pygame.Surface(success_surf.get_size(), pygame.SRCALPHA)
    success_surf_temp.blit(success_shadow, (2, 2))
    success_surf_temp.blit(success_surf, (0, 0))
    success_surf_temp.set_alpha(check_alpha)
    
    screen.blit(success_surf_temp, (WIDTH//2 - success_surf.get_width()//2, scaled_y + 120))
    
    # Efeito de partículas/estrelas ao redor (opcional - adiciona mais dinamismo)
    star_count = 12
    for i in range(star_count):
        angle = (i / star_count) * 2 * math.pi
        radius = 100 + 30 * math.sin(pygame.time.get_ticks() * 0.005 + i)
        star_x = WIDTH//2 + radius * math.cos(angle)
        star_y = HEIGHT//2 + radius * math.sin(angle)
        
        star_size = int(15 * (1 + 0.5 * math.sin(pygame.time.get_ticks() * 0.01 + i)))
        star_alpha = int(alpha * 0.7)
        
        star_surf = pygame.Surface((star_size * 2, star_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(star_surf, (255, 255, 255, star_alpha), (star_size, star_size), star_size)
        screen.blit(star_surf, (star_x - star_size, star_y - star_size))

def draw_guess_modal():
    """Desenha o modal para adivinhar a música"""
    global btn_modal_confirm, btn_modal_cancel
    
    # Overlay escuro semi-transparente
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))
    
    # Card do modal centralizado
    modal_width = 600
    modal_height = 350
    modal_x = (WIDTH - modal_width) // 2
    modal_y = (HEIGHT - modal_height) // 2
    
    modal_rect = draw_card(screen, (modal_x, modal_y, modal_width, modal_height), BG_CARD, gradient=True)
    
    # Borda brilhante no modal
    pygame.draw.rect(screen, ACCENT, modal_rect, width=3, border_radius=20)
    
    # Título do modal
    title_text = FONT_TITLE.render("ADIVINHE A MÚSICA", True, ACCENT)
    title_shadow = FONT_TITLE.render("ADIVINHE A MÚSICA", True, (0, 0, 0))
    screen.blit(title_shadow, (modal_x + modal_width//2 - title_text.get_width()//2 + 2, modal_y + 30 + 2))
    screen.blit(title_text, (modal_x + modal_width//2 - title_text.get_width()//2, modal_y + 30))
    
    # Subtítulo com dica
    hint_text = FONT_SMALL.render("Digite o nome da música que você acha que é:", True, TEXT_SECONDARY)
    screen.blit(hint_text, (modal_x + 40, modal_y + 90))
    
    # Campo de input no modal
    input_y = modal_y + 130
    input_width = modal_width - 80
    input_height = 60
    input_x = modal_x + 40
    
    input_rect = draw_card(screen, (input_x, input_y, input_width, input_height), BG_SURFACE, border_radius=15, gradient=True)
    
    # Borda pulsante quando ativo
    if input_active:
        pulse_time = pygame.time.get_ticks() * 0.01
        pulse_width = int(2 + math.sin(pulse_time) * 1)
        pygame.draw.rect(screen, ACCENT, input_rect, width=pulse_width, border_radius=15)
    else:
        pygame.draw.rect(screen, GRAY_700, input_rect, width=1, border_radius=15)
    
    # Texto do input
    display_text = user_text if (input_active or user_text) else "Digite o nome da música..."
    col = TEXT_PRIMARY if input_active else TEXT_SECONDARY
    text_surf = FONT.render(display_text, True, col)
    
    # Limita o texto visível se for muito longo
    max_width = input_width - 40
    if text_surf.get_width() > max_width:
        # Tenta renderizar com fonte menor
        text_surf = FONT_SMALL.render(display_text, True, col)
        if text_surf.get_width() > max_width:
            # Se ainda for muito longo, trunca visualmente
            display_text_short = display_text[:30] + "..."
            text_surf = FONT_SMALL.render(display_text_short, True, col)
    
    screen.blit(text_surf, (input_rect.x + 20, input_rect.y + (input_height - text_surf.get_height()) // 2))
    
    # Indicador de cursor quando ativo
    if input_active:
        cursor_x = input_rect.x + 20 + text_surf.get_width() + 2
        cursor_time = pygame.time.get_ticks() // 500  # Pisca a cada 500ms
        if cursor_time % 2 == 0:
            pygame.draw.line(screen, TEXT_PRIMARY, (cursor_x, input_rect.y + 15), (cursor_x, input_rect.y + input_height - 15), 2)
    
    # Botões do modal
    btn_width = 200
    btn_height = 55
    btn_spacing = 30
    
    # Botão Confirmar
    confirm_x = modal_x + (modal_width - (btn_width * 2 + btn_spacing)) // 2
    confirm_y = modal_y + modal_height - 90
    
    btn_modal_confirm = Button("CONFIRMAR", (confirm_x, confirm_y, btn_width, btn_height), 
                               color=SUCCESS, hover=SUCCESS_HOVER, font=FONT)
    btn_modal_confirm.draw(screen)
    
    # Botão Cancelar
    cancel_x = confirm_x + btn_width + btn_spacing
    btn_modal_cancel = Button("CANCELAR", (cancel_x, confirm_y, btn_width, btn_height), 
                              color=DANGER, hover=DANGER_HOVER, font=FONT)
    btn_modal_cancel.draw(screen)
    
    # Informação sobre vidas e pontos
    info_text = FONT_TINY.render(f"Acertar: +5 pontos | Errar: -1 vida", True, TEXT_SECONDARY)
    screen.blit(info_text, (modal_x + modal_width//2 - info_text.get_width()//2, modal_y + modal_height - 35))
    
    return btn_modal_confirm, btn_modal_cancel

def draw_play():
    # Background com gradiente roxo-azul
    purple_start = (60, 20, 80)  # Roxo escuro
    blue_end = (20, 40, 100)     # Azul escuro
    draw_gradient(screen, (0, 0, WIDTH, HEIGHT), purple_start, blue_end, vertical=True)

    # Header fixo no topo - estilo game HUD
    header_height = 80
    header_surf = pygame.Surface((WIDTH, header_height), pygame.SRCALPHA)
    
    # Fundo do header com gradiente escuro
    header_grad_start = (30, 20, 50)
    header_grad_end = (20, 15, 35)
    draw_gradient(header_surf, (0, 0, WIDTH, header_height), header_grad_start, header_grad_end, vertical=True)
    
    # Borda inferior brilhante
    pygame.draw.line(header_surf, ACCENT, (0, header_height - 2), (WIDTH, header_height - 2), 2)
    pygame.draw.line(header_surf, (ACCENT[0]//2, ACCENT[1]//2, ACCENT[2]//2), (0, header_height - 1), (WIDTH, header_height - 1), 1)
    
    # Linha divisória central decorativa
    center_x = WIDTH // 2
    pygame.draw.line(header_surf, GRAY_700, (center_x, 10), (center_x, header_height - 10), 1)
    
    screen.blit(header_surf, (0, 0))
    
    # Ícone musical decorativo no centro do header
    music_icon = "♪"
    icon_font = get_font("Montserrat", 40, bold=True)
    icon_surf = icon_font.render(music_icon, True, ACCENT)
    screen.blit(icon_surf, (center_x - icon_surf.get_width()//2, 20))
    
    # Barra de progresso do jogo (notas reveladas)
    if current_song_seq:
        progress_y = 70
        progress_width = WIDTH - 100
        progress_height = 8
        progress_x = 50
        
        # Fundo da barra
        progress_bg_rect = pygame.Rect(progress_x, progress_y, progress_width, progress_height)
        pygame.draw.rect(screen, GRAY_700, progress_bg_rect, border_radius=4)
        
        # Barra de progresso preenchida
        total_notes = len(current_song_seq)
        progress = current_index / total_notes if total_notes > 0 else 0
        progress_fill_width = int(progress_width * progress)
        progress_fill_rect = pygame.Rect(progress_x, progress_y, progress_fill_width, progress_height)
        
        # Preenche a barra com gradiente
        if progress_fill_width > 0:
            progress_color = SUCCESS if progress >= 1.0 else ACCENT
            draw_gradient(screen, progress_fill_rect, progress_color, 
                         tuple(min(255, c + 30) for c in progress_color), vertical=False)
        
        # Borda brilhante na barra
        pygame.draw.rect(screen, ACCENT, progress_bg_rect, width=1, border_radius=4)
        
        # Texto de progresso
        progress_text = f"{current_index}/{total_notes}"
        progress_label = FONT_TINY.render(progress_text, True, TEXT_SECONDARY)
        screen.blit(progress_label, (progress_x + progress_width - progress_label.get_width(), progress_y - 18))

    # Pontuação grande no topo esquerdo - estilo game HUD
    score_label = FONT_SMALL.render("PONTUAÇÃO", True, TEXT_SECONDARY)
    screen.blit(score_label, (30, 10))
    score_text = FONT_TITLE.render(f"{score}", True, (255, 215, 0))  # Dourado para destacar
    # Sombra do texto de pontuação para efeito 3D
    shadow_score = FONT_TITLE.render(f"{score}", True, (0, 0, 0))
    screen.blit(shadow_score, (32, 37))
    screen.blit(score_text, (30, 35))
    # Brilho dourado sutil ao redor da pontuação
    glow_rect = pygame.Rect(25, 30, score_text.get_width() + 10, score_text.get_height() + 10)
    glow_surf = pygame.Surface((glow_rect.w, glow_rect.h), pygame.SRCALPHA)
    pygame.draw.rect(glow_surf, (255, 215, 0, 50), glow_surf.get_rect(), width=2, border_radius=5)
    screen.blit(glow_surf, glow_rect.topleft)
    
    # Vidas no topo direito - com ícones de coração
    hearts_x = WIDTH - 250
    hearts_y = 25
    lives_label = FONT_SMALL.render("VIDAS", True, TEXT_SECONDARY)
    screen.blit(lives_label, (hearts_x, 15))
    
    # Desenha corações para cada vida - estilo game
    heart_size = 28
    for i in range(3):
        heart_x = hearts_x + (i * (heart_size + 15))
        heart_color = DANGER if i < lives else GRAY_600
        # Desenha coração estilo game (mais arredondado)
        heart_center_x = heart_x + heart_size//2
        heart_center_y = hearts_y + 15
        
        # Desenha dois círculos para formar o topo do coração
        top_radius = heart_size // 4
        pygame.draw.circle(screen, heart_color, (heart_center_x - top_radius//2, heart_center_y - top_radius//2), top_radius)
        pygame.draw.circle(screen, heart_color, (heart_center_x + top_radius//2, heart_center_y - top_radius//2), top_radius)
        
        # Desenha triângulo para a parte inferior do coração
        heart_points = [
            (heart_center_x, heart_center_y + top_radius),
            (heart_center_x - top_radius * 1.5, heart_center_y),
            (heart_center_x + top_radius * 1.5, heart_center_y),
        ]
        pygame.draw.polygon(screen, heart_color, heart_points)
        
        # Sombra interna para dar profundidade
        if i < lives:
            inner_color = tuple(min(255, c + 40) for c in heart_color)
            pygame.draw.circle(screen, inner_color, (heart_center_x - top_radius//2 + 2, heart_center_y - top_radius//2 - 2), top_radius - 2)
            pygame.draw.circle(screen, inner_color, (heart_center_x + top_radius//2 - 2, heart_center_y - top_radius//2 - 2), top_radius - 2)
    
    # Texto de vidas também
    lives_text = FONT_HEADING.render(f"x{lives}", True, DANGER if lives > 0 else GRAY_600)
    screen.blit(lives_text, (hearts_x + 110, hearts_y + 5))

    # Card principal de informações com gradiente e borda destacada (estilo game)
    card_main = draw_card(screen, (50, 100, WIDTH-100, 150), BG_CARD, gradient=True)
    # Borda brilhante no card principal
    pygame.draw.rect(screen, ACCENT, card_main, width=2, border_radius=20)

    # Música (oculta ou revelada) - estilo game
    nome_show = current_song_data.nome if state == 'gameover' else '???'
    music_label = FONT_SMALL.render("MÚSICA", True, TEXT_SECONDARY)
    music_name_color = ACCENT if state == 'gameover' else WARNING
    music_name = FONT_HEADING.render(nome_show, True, music_name_color)
    # Sombra no nome da música para efeito 3D
    shadow_music = FONT_HEADING.render(nome_show, True, (0, 0, 0))
    screen.blit(shadow_music, (card_main.x + 32, card_main.y + 52))
    screen.blit(music_label, (card_main.x + 30, card_main.y + 25))
    screen.blit(music_name, (card_main.x + 30, card_main.y + 50))

    # Notas liberadas - estilo game badge
    displayed = [n[0] for n in current_song_seq[:current_index]]
    txt = " • ".join(displayed) if displayed else "Nenhuma nota revelada ainda"
    notes_label = FONT_SMALL.render("NOTAS REVELADAS", True, TEXT_SECONDARY)
    notes_text = FONT.render(txt, True, SUCCESS if displayed else TEXT_SECONDARY)
    screen.blit(notes_label, (card_main.x + 30, card_main.y + 95))
    # Badge para notas reveladas
    notes_badge_rect = pygame.Rect(card_main.x + 30, card_main.y + 120, len(txt) * 12 + 20, 30)
    draw_card(screen, notes_badge_rect, BG_SURFACE, border_radius=15)
    screen.blit(notes_text, (notes_badge_rect.x + 10, notes_badge_rect.y + 5))

    # Próxima nota (se houver) - card destacado estilo game
    global play_here_button
    if current_index < len(current_song_seq):
        card_next = draw_card(screen, (400, 100, 550, 150), BG_SURFACE, gradient=True)
        # Borda pulsante na próxima nota
        pulse_time = pygame.time.get_ticks() * 0.005
        pulse_intensity = 0.7 + 0.3 * math.sin(pulse_time)
        border_color_pulse = tuple(int(c * pulse_intensity) for c in WARNING)
        pygame.draw.rect(screen, border_color_pulse, card_next, width=3, border_radius=20)
        
        next_label = FONT_SMALL.render("PRÓXIMA NOTA", True, TEXT_SECONDARY)
        next_value = FONT_TITLE.render("?", True, WARNING)
        # Sombra na interrogação
        shadow_next = FONT_TITLE.render("?", True, (0, 0, 0))
        screen.blit(shadow_next, (card_next.x + 32, card_next.y + 52))
        screen.blit(next_label, (card_next.x + 30, card_next.y + 25))
        screen.blit(next_value, (card_next.x + 30, card_next.y + 50))

        btn_play_here = Button("Ouvir", (card_next.x + 200, card_next.y + 60, 150, 50), color=WARNING, font=FONT_SMALL)
        btn_play_here.draw(screen)
        play_here_button = btn_play_here

    # Botões de ação - estilo game com ícones
    # Botões de ação
    btn_repeat.draw(screen)
    btn_action_sing.draw(screen)
    btn_guess.draw(screen)

    # Desenha o modal de adivinhar música se estiver aberto
    global btn_modal_confirm, btn_modal_cancel
    if guess_modal_open:
        btn_modal_confirm, btn_modal_cancel = draw_guess_modal()

    # Mensagem de feedback - estilo game notification
    if message:
        msg_color = WARNING if "tempo" in message.lower() else SUCCESS if "acertou" in message.lower() or "perfeito" in message.lower() else DANGER if "errou" in message.lower() else TEXT_PRIMARY
        # Card de notificação
        msg_surf = FONT.render(message, True, msg_color)
        msg_card_rect = pygame.Rect(50, HEIGHT - 80, msg_surf.get_width() + 40, 50)
        msg_bg_color = BG_CARD
        msg_bg_alpha = 200
        if "acertou" in message.lower() or "perfeito" in message.lower():
            msg_bg_color = tuple(min(255, c + 30) for c in SUCCESS)
        elif "errou" in message.lower():
            msg_bg_color = tuple(min(255, c + 30) for c in DANGER)
        elif "tempo" in message.lower():
            msg_bg_color = tuple(min(255, c + 30) for c in WARNING)
        
        msg_card_surf = pygame.Surface((msg_card_rect.w, msg_card_rect.h), pygame.SRCALPHA)
        # Aplica alpha ao fundo
        bg_with_alpha = (*msg_bg_color, msg_bg_alpha)
        pygame.draw.rect(msg_card_surf, bg_with_alpha, msg_card_surf.get_rect(), border_radius=15)
        pygame.draw.rect(msg_card_surf, (*msg_color, 255), msg_card_surf.get_rect(), width=2, border_radius=15)
        screen.blit(msg_card_surf, msg_card_rect.topleft)
        screen.blit(msg_surf, (msg_card_rect.x + 20, msg_card_rect.y + 10))
        msg_color = WARNING if "tempo" in message.lower() else SUCCESS if "acertou" in message.lower() else TEXT_PRIMARY
        msg_surf = FONT_SMALL.render(message, True, msg_color)
        screen.blit(msg_surf, (200, HEIGHT - 50))

    # Botão para retornar ao menu
    btn_menu.draw(screen)
    
    # Desenha a animação de sucesso se ativa
    draw_success_animation()

def draw_detector():
    purple_start = (60, 20, 80)
    blue_end = (20, 40, 100)    
    draw_gradient(screen, (0, 0, WIDTH, HEIGHT), purple_start, blue_end, vertical=True)

    draw_text_with_shadow(screen, "DETECTOR DE PITCH", FONT_SUBTITLE, ACCENT, (50, 30), shadow_offset=3)

    target = current_song_seq[current_index][0] if current_index < len(current_song_seq) else "-"
    card_target = draw_card(screen, (50, 100, WIDTH-350, 200), BG_CARD, gradient=True)

    target_label = FONT_SMALL.render("Cante e SEGURE esta nota:", True, TEXT_SECONDARY)
    screen.blit(target_label, (card_target.x + 30, card_target.y + 25))

    target_surf = FONT_TITLE.render(target, True, WARNING)
    screen.blit(target_surf, (card_target.x + 30, card_target.y + 60))

    instruction = FONT_TINY.render("Mantenha a nota estável por 1 segundo", True, TEXT_SECONDARY)
    screen.blit(instruction, (card_target.x + 30, card_target.y + 160))

    # Card de detecção com gradiente
    card_detect = draw_card(screen, (50, 320, WIDTH-350, 250), BG_SURFACE, gradient=True)

    detect_label = FONT_SMALL.render("Detecção em tempo real:", True, TEXT_SECONDARY)
    screen.blit(detect_label, (card_detect.x + 30, card_detect.y + 25))
    target_freq = NOTE_FREQS.get(target)
    gauge_rect = (card_detect.x + 30, card_detect.y + 120, card_detect.w - 60, 120)

    if detected_name:
        detected_surf = FONT_HEADING.render(detected_name, True, SUCCESS)
        freq_surf = FONT_SMALL.render(f"{detected_freq:.1f} Hz", True, TEXT_SECONDARY)
        screen.blit(detected_surf, (card_detect.x + 30, card_detect.y + 60))
        screen.blit(freq_surf, (card_detect.x + 30, card_detect.y + 95))

        deviation_to_use = detected_deviation_hz if detected_deviation_hz is not None else (detected_freq - target_freq if target_freq else None)

        draw_needle_gauge(
            screen,
            gauge_rect,
            detected_freq,
            target_freq,
            tolerance_hz=30.0,
            min_freq=0.0,
            max_freq=500.0,
        )

    else:
        no_detect = FONT.render("Aguardando entrada...", True, TEXT_SECONDARY)
        screen.blit(no_detect, (card_detect.x + 30, card_detect.y + 60))
        draw_needle_gauge(
            screen,
            gauge_rect,
            None,
            target_freq,
            tolerance_hz=30.0,
            min_freq=0.0,
            max_freq=500.0,
        )

    if target_freq:
        target_text = FONT_TINY.render(f"Alvo: {target} = {target_freq:.1f} Hz", True, TEXT_SECONDARY)
        screen.blit(target_text, (card_detect.x + 30, card_detect.y + 110))

    # Mensagem de status
    msg_y = card_detect.y + 215
    msg_color = WARNING if detector.running else TEXT_SECONDARY
    if detector_result is True: msg_color = SUCCESS
    elif detector_result is False: msg_color = DANGER
    msg_surf = FONT_SMALL.render(message, True, msg_color)
    screen.blit(msg_surf, (card_detect.x + 40, msg_y))

    cooldown_active = time.time() < button_cooldown_until
    if cooldown_active:
        btn_start_listen.color = (150, 150, 150)
        btn_start_listen.hover = (150, 150, 150)   
    else:
        btn_start_listen.color = (0, 200, 0)       
        btn_start_listen.hover = (0, 200, 0)       


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
                    notas_reveladas = current_song_seq[:current_index]
                    for nota_nome, duracao in notas_reveladas:
                        play_note(NOTE_FREQS[nota_nome], duracao, record=False)
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
                guess_modal_open = True
                input_active = True
                user_text = ""

            # Processa eventos do modal de adivinhar música
            if guess_modal_open:
                # Fecha o modal se clicar fora dele (no overlay)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    modal_width = 600
                    modal_height = 350
                    modal_x = (WIDTH - modal_width) // 2
                    modal_y = (HEIGHT - modal_height) // 2
                    modal_rect = pygame.Rect(modal_x, modal_y, modal_width, modal_height)
                    if not modal_rect.collidepoint(event.pos):
                        # Clicou fora do modal, fecha
                        guess_modal_open = False
                        input_active = False
                        user_text = ""
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Fecha o modal ao pressionar ESC
                        guess_modal_open = False
                        input_active = False
                        user_text = ""
                    elif event.key == pygame.K_RETURN:
                        # Processa o palpite ao pressionar ENTER
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
                        guess_modal_open = False
                    elif event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]
                    else:
                        if len(user_text) < 40: 
                            user_text += event.unicode
                
                # Verifica cliques nos botões do modal
                if btn_modal_confirm and btn_modal_confirm.clicked(event):
                    # Processa o palpite
                    guess = user_text.strip()
                    real = current_song_data.nome or ""

                    if is_similar_enough(guess, real):
                        score += 5
                        similarity = calculate_similarity(guess, real)

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
                        message = f"✗ Errou! Vidas: {lives}"
                        if lives <= 0:
                            state = 'gameover'
                    
                    user_text = ""
                    input_active = False
                    guess_modal_open = False
                
                if btn_modal_cancel and btn_modal_cancel.clicked(event):
                    # Fecha o modal sem processar
                    guess_modal_open = False
                    input_active = False
                    user_text = ""


        elif state == 'detector':
            if btn_back.clicked(event):
                detector.stop()
                state = 'play'

            if btn_play_target.clicked(event):
                if current_index < len(current_song_seq):
                    n = current_song_seq[current_index]
                    threading.Thread(target=play_note, args=(NOTE_FREQS[n[0]], n[1]), daemon=True).start()

            cooldown_active = time.time() < button_cooldown_until

            if not cooldown_active and btn_start_listen.clicked(event):
                if current_index < len(current_song_seq):
                    target_name = current_song_seq[current_index][0]
                    start_detector_thread(target_name)

                button_cooldown_until = time.time() + 10

            if btn_skip_confirm.clicked(event):
                if detector_result is True:
                    # Ativa a animação de sucesso
                    show_success_animation = True
                    success_animation_start_time = pygame.time.get_ticks()
                    
                    current_index += 1
                    message = "Nota desbloqueada!"
                    state = 'play'
                else:
                    message = "Segure a nota por 1s até aparecer ACERTOU."

        elif state == 'gameover':
            # Calcula as mesmas coordenadas usadas no desenho
            card_y = HEIGHT//2 - 250
            btn_y1 = card_y + 390
            btn_y2 = card_y + 465
            
            # Verifica cliques nos botões do game over
            btn_play_again = Button("JOGAR NOVAMENTE", (WIDTH//2 - 150, btn_y1, 300, 60), 
                                   color=SUCCESS, hover=SUCCESS_HOVER, font=FONT_HEADING)
            btn_menu_gameover = Button("MENU PRINCIPAL", (WIDTH//2 - 150, btn_y2, 300, 60), 
                                      color=ACCENT, hover=ACCENT_HOVER, font=FONT_HEADING)
            
            if btn_play_again.clicked(event):
                lives = 3
                score = 0
                start_round()
                if current_song_seq:
                    primeira_nota = current_song_seq[0]
                    threading.Thread(target=play_note, args=(NOTE_FREQS[primeira_nota[0]], primeira_nota[1]), daemon=True).start()
                state = 'play'
            if btn_menu_gameover.clicked(event):
                state = 'menu'

    if state == 'menu': draw_menu()
    elif state == 'rules': draw_rules()
    elif state == 'settings': draw_settings()
    elif state == 'play': draw_play()
    elif state == 'detector': draw_detector()
    elif state == 'gameover':
        # Background com gradiente roxo-azul
        purple_start = (60, 20, 80)  # Roxo escuro
        blue_end = (20, 40, 100)     # Azul escuro
        draw_gradient(screen, (0, 0, WIDTH, HEIGHT), purple_start, blue_end, vertical=True)

        # Card central de game over com gradiente
        card = draw_card(screen, (WIDTH//2 - 300, HEIGHT//2 - 250, 600, 500), BG_CARD, gradient=True)
        
        # Borda brilhante no card
        pygame.draw.rect(screen, DANGER, card, width=3, border_radius=20)

        # Título com sombra
        title_surf = FONT_TITLE.render("FIM DE JOGO", True, DANGER)
        title_shadow = FONT_TITLE.render("FIM DE JOGO", True, (0, 0, 0))
        screen.blit(title_shadow, (WIDTH//2 - title_surf.get_width()//2 + 2, card.y + 52))
        screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, card.y + 50))

        # Pontuação com destaque - estilo game
        score_label = FONT_SMALL.render("PONTUAÇÃO FINAL", True, TEXT_SECONDARY)
        screen.blit(score_label, (WIDTH//2 - score_label.get_width()//2, card.y + 130))

        # Card para pontuação
        score_card = draw_card(screen, (WIDTH//2 - 150, card.y + 160, 300, 80), BG_SURFACE, border_radius=15, gradient=True)
        pygame.draw.rect(screen, (255, 215, 0), score_card, width=2, border_radius=15)
        
        score_surf = FONT_TITLE.render(f"{score}", True, (255, 215, 0))  # Dourado
        score_shadow = FONT_TITLE.render(f"{score}", True, (0, 0, 0))
        screen.blit(score_shadow, (WIDTH//2 - score_surf.get_width()//2 + 2, card.y + 187))
        screen.blit(score_surf, (WIDTH//2 - score_surf.get_width()//2, card.y + 185))
        
        # Pontos ou Ponto (singular/plural)
        pontos_text = "pontos" if score != 1 else "ponto"
        pontos_surf = FONT_SMALL.render(pontos_text, True, TEXT_SECONDARY)
        screen.blit(pontos_surf, (WIDTH//2 - pontos_surf.get_width()//2, card.y + 245))

        # Música revelada
        music_name = current_song_data.nome if current_song_data else "Desconhecida"
        music_label = FONT_SMALL.render("A MÚSICA ERA", True, TEXT_SECONDARY)
        screen.blit(music_label, (WIDTH//2 - music_label.get_width()//2, card.y + 280))
        
        # Card para nome da música
        music_card = draw_card(screen, (WIDTH//2 - 200, card.y + 310, 400, 50), BG_SURFACE, border_radius=15)
        music_surf = FONT_HEADING.render(music_name, True, WARNING)
        screen.blit(music_surf, (WIDTH//2 - music_surf.get_width()//2, card.y + 320))

        # Botões de ação - estilo game
        btn_play_again = Button("JOGAR NOVAMENTE", (WIDTH//2 - 150, card.y + 390, 300, 60), 
                               color=SUCCESS, hover=SUCCESS_HOVER, font=FONT_HEADING)
        btn_menu_gameover = Button("MENU PRINCIPAL", (WIDTH//2 - 150, card.y + 465, 300, 60), 
                                  color=ACCENT, hover=ACCENT_HOVER, font=FONT_HEADING)
        
        btn_play_again.draw(screen)
        btn_menu_gameover.draw(screen)

    pygame.display.flip()
    CLOCK.tick(30)

pygame.quit()