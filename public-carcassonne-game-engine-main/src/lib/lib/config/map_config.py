from typing import Callable

from dotmap import DotMap


MAX_MAP_LENGTH = 169
MAP_CENTER = (85, 85)
MONASTARY_IDENTIFIER = "MONASTARY"

tile_counts = DotMap(
    {
        "A": 2,
        "B": 4,
        "C": 1,
        "D": 4,
        "E": 5,
        "F": 2,
        "G": 1,
        "H": 3,
        "I": 2,
        "J": 3,
        "K": 3,
        "L": 3,
        "M": 2,
        "N": 3,
        "O": 2,
        "P": 3,
        "Q": 1,
        "R": 3,
        "S": 2,
        "T": 1,
        "U": 8,
        "V": 9,
        "W": 4,
        "X": 1,
        "RS": 1,
        "R1": 1,
        "R2": 2,
        "R3": 1,
        "R4": 1,
        "R5": 1,
        "R6": 1,
        "R7": 1,
        "R8": 1,
        "R9": 1,
        "RE": 1,
    },
    _dynamic=False,
)

NUM_PLACEABLE_TILE_TYPES = 9

TILE_EDGE_IDS: dict[str, int] = {
    "top_edge": 0,
    "right_edge": 1,
    "bottom_edge": 2,
    "left_edge": 3,
}

TILE_EXTERNAL_POS: dict[str, Callable[[int, int], tuple[int, int]]] = {
    "top_edge": lambda x, y: (x, y - 1),
    "right_edge": lambda x, y: (x + 1, y),
    "bottom_edge": lambda x, y: (x, y + 1),
    "left_edge": lambda x, y: (x - 1, y - 1),
}
