__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import pytest
from utils.distance import character_distance_between_strings


def test_character_distance_between_strings():

    # Test equal strings with character 'a' and 'c' at the same positions
    assert character_distance_between_strings("abc", "cbc") == 2

    # Test equal strings with character 'a' and 'c' at different positions
    assert character_distance_between_strings("abc", "cba") == 4

    # Test unequal strings
    with pytest.raises(ValueError):
        character_distance_between_strings("abc", "abcd")



