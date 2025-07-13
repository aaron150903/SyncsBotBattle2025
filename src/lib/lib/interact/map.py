from collections import defaultdict
from lib.interact.tile import (
    Tile,
    create_base_tiles,
    create_river_tiles,
    # create_expansion_tiles,
)

from lib.config.map_config import MAX_MAP_LENGTH


class Map:
    def __init__(self) -> None:
        self.placed_tiles: list[Tile] = []
        self.available_tiles: set[Tile] = set()
        self.available_tiles_by_type: dict[str, list[Tile]] = defaultdict(list)

        self._grid: list[list[Tile | None]] = [
            [None for _ in range(MAX_MAP_LENGTH)] for _ in range(MAX_MAP_LENGTH)
        ]

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
