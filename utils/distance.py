"""To evaluate the distance between two strings"""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"


def _character_distance(char1: str, char2: str) -> int:
    """
    Calculates the distance between two characters.

    Args:
        char1 (str): The first character.
        char2 (str): The second character.

    Returns:
        int: The distance between the two characters.
    """
    return abs(ord(char1) - ord(char2))


def character_distance_between_strings(s1: str, s2: str) -> int:
    """
    Calculates the character-wise distance between two strings of the same length.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        int: The character-wise distance between the two strings.

    Raises:
        ValueError: If the input strings have different lengths.
    """
    if len(s1) != len(s2):
        raise ValueError("Strings must be of the same length")

    total_distance = 0

    for char1, char2 in zip(s1, s2):
        total_distance += _character_distance(char1, char2)

    return total_distance
