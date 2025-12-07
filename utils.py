from difflib import SequenceMatcher

def calculate_similarity(text1, text2):
    """
    Calcula a similaridade entre duas strings (0.0 a 1.0).
    Retorna um valor entre 0 (completamente diferente) e 1 (idêntico).
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def is_similar_enough(guess, target, threshold=0.75):
    """
    Verifica se o texto adivinhado é similar o suficiente ao alvo.

    Args:
        guess: Texto digitado pelo usuário
        target: Texto correto (nome da música)
        threshold: Nível de similaridade necessário (0.0 a 1.0)

    Returns:
        True se for similar o suficiente, False caso contrário
    """
    guess_normalized = ' '.join(guess.lower().split())
    target_normalized = ' '.join(target.lower().split())

    similarity = calculate_similarity(guess_normalized, target_normalized)

    contains = guess_normalized in target_normalized and len(guess_normalized) > 3

    return similarity >= threshold or contains