from dataclasses import dataclass
from typing import List, Tuple

# Definição da "Struct"
@dataclass
class Musica:
    nome: str
    genero: str
    # Uma lista de tuplas, onde cada tupla é (Nota, Duração em segundos)
    notas: List[Tuple[str, float]] 

# ==========================================
# BANCO DE DADOS DE MÚSICAS
# ==========================================

BIBLIOTECA = [
    Musica(
        nome="Brilha Brilha Estrelinha",
        genero="Infantil",
        notas=[
            ("D", 0.5), ("D", 0.5), ("A", 0.5), ("A", 0.5), 
            ("B", 0.5), ("B", 0.5), ("A", 1.0), # <--- Nota final mais longa
            ("G", 0.5), ("G", 0.5), ("F#", 0.5), ("F#", 0.5), 
            ("E", 0.5), ("E", 0.5), ("D", 1.0)
        ]
    ),
    Musica(
        nome="Parabéns pra Você",
        genero="Festas",
        notas=[
            ("D", 0.3), ("D", 0.3), ("E", 0.8), ("D", 0.8), 
            ("G", 0.8), ("F#", 1.5)
        ]
    ),
    Musica(
        nome="Ode à Alegria (Beethoven)",
        genero="Clássica",
        notas=[
            ("F#", 0.5), ("F#", 0.5), ("G", 0.5), ("A", 0.5),
            ("A", 0.5), ("G", 0.5), ("F#", 0.5), ("E", 0.5),
            ("D", 0.5), ("D", 0.5), ("E", 0.5), ("F#", 0.5),
            ("F#", 0.7), ("E", 0.2), ("E", 1.0) 
        ]
    )
]