from collections import defaultdict
from lib.interact.tile import (
    Tile,
    create_base_tiles,
    create_river_tiles,
    # create_expansion_tiles,
)

from lib.config.map_config import MAX_MAP_LENGTH
from lib.interact.structure import StructureType


class Map:
    def __init__(self) -> None:
        self.placed_tiles: list[Tile] = []
        self.available_tiles: set[Tile] = set()
        self.available_tiles_by_type: dict[str, list[Tile]] = defaultdict(list)

        self._grid: list[list[Tile | None]] = [
            [None for _ in range(MAX_MAP_LENGTH)] for _ in range(MAX_MAP_LENGTH)
        ]
        self.straight_rivers: int = 6

    def start_base_phase(self) -> None:
        assert not self.available_tiles
        self.available_tiles.update(set(create_base_tiles()))

        for tile in self.available_tiles:
            self.available_tiles_by_type[tile.tile_type].append(tile)

    def start_river_phase(self) -> None:
        assert not self.available_tiles

        self.available_tiles.update(set(create_river_tiles()))

        for tile in self.available_tiles:
            self.available_tiles_by_type[tile.tile_type].append(tile)

    def place_river_start(self, pos: tuple[int, int]) -> None:
        starting_tile = Tile.get_starting_tile()
        starting_tile.placed_pos = pos

        self._grid[pos[1]][pos[0]] = starting_tile
        self.placed_tiles.append(starting_tile)

    def place_river_end(self, pos: tuple[int, int], rotation: int) -> None:
        river_end_tile = Tile.get_river_end_tile()
        river_end_tile.rotate_clockwise(rotation)
        river_end_tile.placed_pos = pos

        self._grid[pos[1]][pos[0]] = river_end_tile
        self.placed_tiles.append(river_end_tile)

    def add_expansion_pack(self, expansion_pack: None) -> None:
        pass

    def get_tile_by_type(self, type: str, pop: bool) -> "Tile":
        if pop:
            tile = self.available_tiles_by_type[type].pop()
            self.available_tiles.remove(tile)
            return tile

        return self.available_tiles_by_type[type][0]

    # Returns if a river tile can be placed at position (x,y)
    def river_validation(self, tile: Tile, x: int, y: int) -> str:
        neighbouring_tiles = {
            edge: Tile.get_external_tile(edge, (x, y), self._grid)
            for edge in Tile.get_edges()
        }
        river_connections = 0
        print("from river validation: ", self.straight_rivers, flush=True)
        for edge, neighbour_tile in neighbouring_tiles.items():
            edge_structure = tile.internal_edges[edge]

            if neighbour_tile:
                # Check if we successfully connected a river structure
                if edge_structure == StructureType.RIVER:
                    river_connections += 1
                    assert river_connections <= 1

            # U - Turn handling -> Handling the case where the edge does not have a tile next to it
            elif edge_structure == StructureType.RIVER:
                # Handling direct u-turns: propagate one out from the proposed disconnected river edge, and check surroundings
                forcast_coordinates = {
                    "top_edge": (0, -1),
                    "right_edge": (1, 0),
                    "bottom_edge": (0, 1),
                    "left_edge": (-1, 0),
                }

                extension = forcast_coordinates[edge]
                forecast_x = x + extension[0]
                forecast_y = y + extension[1]

                for i in range(1, self.straight_rivers + 1):
                    extension = forcast_coordinates[edge]

                    # Look at the tile i tiles away from the direction the river is facing on our current tile
                    forecast_x = x + extension[0] * i
                    forecast_y = y + extension[1] * i

                    for coords in forcast_coordinates.values():
                        checking_x = forecast_x + coords[0]
                        checking_y = forecast_y + coords[1]
                        if not (checking_x == x and checking_y == y):
                            if self._grid[checking_y][checking_x] is not None:
                                return "uturn"

        # Check if there is at least one river edge that is connected
        if river_connections == 0:
            return "disjoint"
        return "pass"
