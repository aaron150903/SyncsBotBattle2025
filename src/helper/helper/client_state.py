from lib.config.map_config import MAX_MAP_LENGTH, MONASTARY_IDENTIFIER
from lib.game.game_logic import GameLogic
from lib.interact.meeple import Meeple
from lib.interact.map import Map
from lib.interact.tile import Tile, TileModifier
from lib.interface.events.typing import EventType
from lib.models.player_model import PlayerModel, PublicPlayerModel
from lib.models.tile_model import TileModel
from lib.interact.structure import StructureType

from copy import deepcopy


class ClientSate(GameLogic):
    def __init__(self) -> None:
        self.round = -1
        self.players: dict[int, PublicPlayerModel]
        self.players_meeples: dict[int, int]
        self.map = Map()

        self.game_over = False
        self.num_placed_tiles = 0

        self.event_history: list["EventType"] = []
        self.new_events: int = 0
        self.turn_order: list[int] = []

        self.points = 0
        self.me: PlayerModel
        self.my_tiles: list[Tile] = []

    def get_meeples_placed_by(self, player_id: int | None) -> list[Meeple]:
        """
        Get Meeples Placed
        Giving None as player id retruns all placed meeples
        """
        meeples = []
        for x in range(MAX_MAP_LENGTH):
            for y in range(MAX_MAP_LENGTH):
                tile = self.map._grid[y][x]

                if tile is None:
                    continue

                for edge, meeple in tile.internal_claims.items():
                    if meeple and (meeple.player_id == player_id or player_id is None):
                        meeples.append(meeple)

        return meeples

    def get_tile_structures(self, tile: TileModel) -> dict[str, StructureType]:
        # Does not return monastary
        found_tile: Tile | None = None
        for t in self.map.available_tiles.union(set(self.map.placed_tiles)):
            if tile.tile_type == t.tile_type:
                found_tile = t

        assert found_tile
        found_tile = deepcopy(found_tile)

        while found_tile.rotation != tile.rotation:
            found_tile.rotate_clockwise(1)

        return {
            edge: structure for edge, structure in found_tile.internal_edges.items()
        }

    def get_placeable_structures(self, my_tile: TileModel) -> dict[str, StructureType]:
        placable_structures: dict[str, StructureType] = {
            e: s
            for e, s in self.get_tile_structures(my_tile).items()
            if StructureType.can_claim(s)
        }

        print(placable_structures, flush=True)

        x, y = my_tile.pos
        tile = self.map._grid[y][x]

        assert tile is not None

        if TileModifier.MONASTARY in tile.modifiers:
            placable_structures[MONASTARY_IDENTIFIER] = StructureType.MONASTARY

        # EP farm expansion

        return placable_structures
