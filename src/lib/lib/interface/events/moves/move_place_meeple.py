from lib.interface.events.moves.base_move import BaseMove
from lib.models.tile_model import TileModel

from typing import Literal


class MovePlaceMeeple(BaseMove):
    event_type: Literal["move_place_meeple"] = "move_place_meeple"
    player_id: int
    tile: TileModel
    placed_on: str  # EDGES or MONASTARY


class MovePlaceMeeplePass(BaseMove):
    event_type: Literal["move_place_meeple_pass"] = "move_place_meeple_pass"
    player_id: int
