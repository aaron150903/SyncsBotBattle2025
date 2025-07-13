from lib.interface.events.moves.base_move import BaseMove
from lib.models.tile_model import TileModel

from typing import Literal


class MovePlaceTile(BaseMove):
    event_type: Literal["move_place_tile"] = "move_place_tile"
    player_id: int
    tile: TileModel
    player_tile_index: int

    def get_public(self) -> "PublicMovePlaceTile":
        return PublicMovePlaceTile(player_id=self.player_id, tile=self.tile)


class PublicMovePlaceTile(BaseMove):
    event_type: Literal["public_move_place_tile"] = "public_move_place_tile"
    player_id: int
    tile: TileModel
