# 🎵 Jogo de Solfejo - Computação Musical

Um jogo educacional de solfejo que utiliza detecção de pitch em tempo real para ensinar reconhecimento de notas musicais. O jogador ouve melodias famosas e deve cantar as notas corretas para progredir no jogo.

## 📋 Índice

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
  - [Windows](#windows)
  - [Linux/Mac](#linuxmac)
- [Como Executar](#como-executar)
- [Como Jogar](#como-jogar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Solução de Problemas](#solução-de-problemas)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)

## ✨ Características

- 🎤 **Detecção de pitch em tempo real** usando aubio
- 🎹 **Sintetizador de piano integrado** com proteção anti-distorção
- 🎮 **Sistema de pontuação e vidas**
- 📚 **Biblioteca de músicas** (Brilha Brilha Estrelinha, Parabéns pra Você, Ode à Alegria)
- 🎯 **Dois modos de jogo**: cantar notas ou adivinhar a música
- 🔊 **Interface gráfica** desenvolvida com Pygame

## 📦 Requisitos

- **Python 3.12** (recomendado) ou Python 3.8+
- **Microfone** funcional conectado ao computador
- **Sistema operacional**: Windows, Linux ou macOS

## 🔧 Instalação

### Windows

1. **Clone ou baixe o projeto:**
```bash
git clone https://github.com/Daniel-Nas/Jogo-Solfejo---Computa-o-Musical---IF754
cd Jogo-Solfejo---Computa-o-Musical---IF754
```

2. **Crie um ambiente virtual:**
```bash
python -m venv venv
```

3. **Ative o ambiente virtual:**
```bash
venv\Scripts\activate
```

4. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

5. **Instale o aubio (Windows):**
```bash
pip install aubio-0.4.9-cp312-cp312-win_amd64.whl
```

### Linux/Mac

1. **Clone ou baixe o projeto:**
```bash
git clone <url-do-repositorio>
cd Jogo-Solfejo---Computa-o-Musical---IF754
```

2. **Crie um ambiente virtual:**
```bash
python3 -m venv venv
```

3. **Ative o ambiente virtual:**
```bash
source venv/bin/activate
```

4. **Instale as dependências do sistema (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev
```

**Para macOS:**
```bash
brew install portaudio
```

5. **Instale as dependências Python:**
```bash
pip install -r requirements.txt
```

## 🚀 Como Executar

1. **Certifique-se de que o ambiente virtual está ativado:**
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

2. **Execute o jogo:**
```bash
python game.py
```

3. **Permita o acesso ao microfone** quando solicitado pelo sistema operacional.

## 🎮 Como Jogar

### Menu Principal
- **INICIAR**: Começa uma nova partida
- **REGRAS**: Mostra as instruções do jogo
- **CONFIGURAÇÕES**: Ajustes de afinação

### Modo de Jogo

1. **Ouça a primeira nota** da melodia (toca automaticamente)

2. **Escolha uma ação:**
   - **Repetir Notas**: Ouve novamente todas as notas já desbloqueadas
   - **🎤 CANTAR NOTA**: Abre o detector de pitch para cantar a próxima nota
   - **TENTAR ADIVINHAR**: Digite o nome da música para ganhar pontos extras

### Detector de Pitch

1. Clique em **"Ouvir Nota Alvo"** para escutar a nota que você precisa cantar
2. Clique em **"Gravar (Microfone)"** para começar a detecção
3. **Cante e SEGURE a nota** por pelo menos **1 segundo**
4. Quando aparecer "ACERTOU!", clique em **"Confirmar e Voltar"**
5. A nota será desbloqueada e você poderá ouvir a sequência completa

### Sistema de Pontuação

- **Vidas**: Você começa com 3 vidas
- **Errar o nome da música**: Perde 1 vida
- **Acertar a música**: Ganha 5 pontos
- **Game Over**: Quando as vidas acabam

## 📁 Estrutura do Projeto

```
Jogo-Solfejo---Computa-o-Musical---IF754/
│
├── game.py                    # Arquivo principal do jogo
├── Musicas.py                 # Banco de dados de músicas
├── README.md                  # Este arquivo
├── aubio-0.4.9-cp312-*.whl   # Biblioteca aubio para Windows
├── venv/                      # Ambiente virtual (criar localmente)
└── .gitignore                 # Arquivos ignorados pelo git
```

### Componentes Principais (game.py)

- **PitchDetector**: Classe para detecção de notas em tempo real
- **Sintetizador de Piano**: Gera sons de piano com harmônicos
- **Sistema de UI**: Menus, botões e interface gráfica
- **Loop Principal**: Gerencia estados do jogo e eventos

### Banco de Músicas (Musicas.py)

Define a estrutura `Musica` com:
- `nome`: Nome da música
- `genero`: Gênero musical
- `notas`: Lista de tuplas (Nota, Duração)

## 🔧 Solução de Problemas

### O microfone não está sendo detectado

**Windows:**
```bash
python -m pyaudio
```
Isso mostrará todos os dispositivos de áudio disponíveis.

**Linux:**
```bash
sudo apt-get install pavucontrol
pavucontrol
```
Verifique se o microfone está habilitado.

### Erro: "No module named 'aubio'"

**Windows:** Certifique-se de instalar o arquivo `.whl` fornecido:
```bash
pip install aubio-0.4.9-cp312-cp312-win_amd64.whl
```

**Linux/Mac:**
```bash
pip install aubio
```

### Erro: "No module named 'pyaudio'"

Instale as dependências do sistema primeiro:

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-dev
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

### O jogo não detecta minha voz

1. Verifique se o microfone está funcionando em outros aplicativos
2. Aproxime-se do microfone
3. Cante mais alto e **segure a nota** por pelo menos 1 segundo
4. Tente cantar na mesma oitava da nota de referência

### Som distorcido ou com ruídos

O jogo já inclui proteção anti-clipping, mas se houver problemas:
1. Reduza o volume do sistema
2. Ajuste o parâmetro `volume` na função `synth_piano_note()` (linha 153 em game.py)

## 🛠️ Tecnologias Utilizadas

- **Python 3.12**: Linguagem principal
- **Pygame**: Interface gráfica e síntese de áudio
- **NumPy**: Processamento numérico e geração de ondas
- **PyAudio**: Captura de áudio do microfone
- **Aubio**: Detecção de pitch (frequência fundamental)

## 📝 Configurações Avançadas

### Ajuste de Afinação

No arquivo `game.py`, você pode ajustar:

```python
A4_TUNING = 440.0       # Frequência do Lá 4 (padrão 440 Hz)
TUNING_OFFSET = 0       # Offset em semitons
REQUIRED_STABILITY = 1.0 # Tempo para segurar a nota (segundos)
LISTEN_DURATION = 10.0  # Tempo máximo de escuta (segundos)
```

### Adicionar Novas Músicas

Edite o arquivo `Musicas.py`:

```python
Musica(
    nome="Nome da Música",
    genero="Gênero",
    notas=[
        ("C", 0.5),  # Dó por 0.5 segundos
        ("D", 0.5),  # Ré por 0.5 segundos
        ("E", 1.0),  # Mi por 1.0 segundo
    ]
)
```

## 👥 Contribuindo

Este projeto foi desenvolvido para a disciplina IF754 - Computação Musical. Contribuições são bem-vindas!

## 📄 Licença

Este projeto é desenvolvido para fins educacionais.

---

**Desenvolvido com ❤️ para IF754 - Computação Musical**
