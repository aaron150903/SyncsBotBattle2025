from typing import TYPE_CHECKING
from lib.config.expansion import EXPANSION

if TYPE_CHECKING:
    from lib.interact.tile import Tile


class Meeple:
    """
    Meeple
    """

    def __init__(self, player_id: int, is_special: bool = False) -> None:
        if not EXPANSION:
            assert not is_special

        self.is_special = is_special
        self.player_id = player_id
        self.placed: Tile | None = None
        self.placed_edge: str = ""

    def _place_meeple(self, tile: "Tile", edge: str) -> None:
        self.placed = tile
        self.placed_edge = edge
        tile.internal_claims[edge] = self

    def _free_meeple(self) -> None:
        assert self.placed
        self.placed.internal_claims[self.placed_edge] = None
        self.placed = None
