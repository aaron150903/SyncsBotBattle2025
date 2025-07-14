from copy import deepcopy
from typing import TYPE_CHECKING

from helper.utils import print_map
from lib.game.game_logic import TileModifier
from engine.config.game_config import MAX_NUM_TILES_IN_HAND
from lib.config.map_config import MONASTARY_IDENTIFIER, NUM_PLACEABLE_TILE_TYPES
from lib.interface.events.moves.move_place_meeple import (
    MovePlaceMeeple,
    MovePlaceMeeplePass,
)
from lib.interface.events.moves.move_place_tile import MovePlaceTile
from lib.interface.events.moves.typing import MoveType
from lib.interface.queries.base_query import BaseQuery
from lib.interact.tile import Tile
from lib.interact.structure import StructureType

import string

VALID_PLACEABLE_TILE_TYPES = [f"R{i}" for i in range(1, NUM_PLACEABLE_TILE_TYPES + 1)]
VALID_PLACEABLE_TILE_TYPES.extend(
    string.ascii_uppercase[: string.ascii_uppercase.index("X") + 1]
)

VALID_ROTATIONS = [0, 1, 2, 3]
VALID_MEEPLE_PLACEMENTS = Tile.get_starting_tile().internal_claims.keys()
VALID_STRUCTURE_CLAIMS = [
    StructureType.MONASTARY,
    StructureType.CITY,
    StructureType.ROAD,
    StructureType.ROAD_START,
]

if TYPE_CHECKING:
    from engine.state.game_state import GameState


class MoveValidator:
    def __init__(self, state: "GameState"):
        self.state = state

    def validate(self, event: MoveType, query: BaseQuery, player_id: int) -> None:
        self._validate_move(event, query, player_id)

        match event:
            case MovePlaceTile() as e:
                self._validate_place_tile(e, query, player_id)
            case MovePlaceMeeple() as e:
                self._validate_place_meeple(e, query, player_id)
            case MovePlaceMeeplePass() as e:
                self._validate_place_meeple_pass(e, query, player_id)

    def _validate_move(self, e: MoveType, query: BaseQuery, player_id: int) -> None:
        if not e.player_id == player_id:
            raise ValueError(
                "You set the move 'player_id' to a player_id other than your own."
            )

    def _validate_place_tile(
        self, e: MovePlaceTile, query: BaseQuery, player_id: int
    ) -> None:
        x, y = e.tile.pos
        tile: Tile

        # R3
        print("Validator recieved tile type", e.tile.tile_type)

        print_map(self.state.map._grid, range(75, 96))

        neighbouring_tiles = {
            edge: Tile.get_external_tile(edge, (x, y), self.state.map._grid)
            for edge in Tile.get_edges()
        }

        # Validate Tile Type
        if e.tile.tile_type not in VALID_PLACEABLE_TILE_TYPES:
            raise ValueError(
                f"You tried placing an invalid tile type - Recieved TileType {e.tile.tile_type}"
            )

        if e.player_tile_index not in range(0, MAX_NUM_TILES_IN_HAND):
            raise ValueError(
                f"You tried placing a tile not in your hand - Incorrect Recieved Tile Index {e.player_tile_index}"
            )

        player = self.state.players[player_id]
        if not any(tile.tile_type == e.tile.tile_type for tile in player.tiles):
            raise ValueError(
                f"You tried placing a tile not in your hand - Tile Type Not in Hand {e.tile.tile_type}"
            )

        tile = self.state.players[player_id].tiles[e.player_tile_index]

        if tile.tile_type != e.tile.tile_type:
            raise ValueError(
                f"You tried placing a tile in your hand but the player tile index mismatched - Player tile index {e.player_tile_index}, Tile Type Given {tile.tile_type}"
            )

        tile = deepcopy(tile)

        # Validate rotation
        if e.tile.rotation not in VALID_ROTATIONS:
            raise ValueError(
                f"You tried placing with an invalid rotation - Recieved Tile Rotation {e.tile.rotation}"
            )

        while tile.rotation != e.tile.rotation:
            tile.rotate_clockwise(1)
        # Validate Tile Pos
        if not any(neighbouring_tiles.values()):
            raise ValueError(
                f"You placed a tile in an empty space - no neighbours at {x, y}"
            )

        # Validating each edge is alighed with a corrrect structure
        river_flag = False
        river_connections = 0

        for edge, neighbour_tile in neighbouring_tiles.items():
            edge_structure = tile.internal_edges[edge]

            # Flag if there is an edge with a river on this tile.
            river_flag = edge_structure == StructureType.RIVER

            if neighbour_tile:
                # Check if edges are aligned with correct structures
                neighboring_edge = neighbour_tile.internal_edges[
                    Tile.get_opposite(edge)
                ]
                if neighboring_edge != edge_structure:
                    # print(tile.tile_type, tile.rotation)
                    # print(neighbour_tile.tile_type, neighbour_tile.rotation)
                    raise ValueError(
                        f"You placed a tile in an mismatched position - {edge} mismatch, your edge is {tile.internal_edges[edge]} on rotation {tile.rotation} at coordinates {e.tile.pos} != {neighbour_tile.internal_edges[Tile.get_opposite(edge)]} on rotation {neighbour_tile.rotation} at position {neighbour_tile.placed_pos}"
                    )

                # Check if we successfully connected a river structure
                if edge_structure == StructureType.RIVER:
                    river_connections += 1
                    assert river_connections <= 1

            # Handling the case where the edge does not have a tile next to it
            # U - Turn handling
            elif edge_structure == StructureType.RIVER:
                # Handling direct u-turns: propagate one out from the proposed disconnected river edge, and check surroundings

                forcast_coordinates_one = {
                    "top_edge": (0, -1),
                    "right_edge": (1, 0),
                    "bottom_edge": (0, 1),
                    "left_edge": (-1, 0),
                }

                extension = forcast_coordinates_one[edge]
                forecast_x = x + extension[0]
                forecast_y = y + extension[1]

                for coords in forcast_coordinates_one.values():
                    checking_x = forecast_x + coords[0]
                    checking_y = forecast_y + coords[1]
                    if not (checking_x == x and checking_y == y):
                        if self.state.map._grid[checking_y][checking_x] is not None:
                            raise ValueError(
                                "You placed a tile that will lead to a U-Turn in the river."
                            )

                # Handling problematic u-turn: if there is two tile away from a disconnected river edge, it means a u-turn has occurred
                forcast_coordinates_two = {
                    "top_edge": (0, -2),
                    "right_edge": (2, 0),
                    "bottom_edge": (0, 2),
                    "left_edge": (-2, 0),
                }
                extension = forcast_coordinates_two[edge]

                # Look at the tile two tiles away from the direction the river is facing on our current tile
                forecast_x = x + extension[0]
                forecast_y = y + extension[1]
                for coords in forcast_coordinates_one.values():
                    checking_x = forecast_x + coords[0]
                    checking_y = forecast_y + coords[1]

                    if self.state.map._grid[checking_y][checking_x] is not None:
                        raise ValueError(
                            "You placed a tile that will lead to a U-Turn in the river."
                        )
        # Check if there is at least one river edge that is connected
        if river_flag and river_connections == 0:
            raise ValueError(
                "You placed a river tile without connecting it to the rest of the river."
            )

    def _validate_place_meeple(
        self, e: MovePlaceMeeple, query: BaseQuery, player_id: int
    ) -> None:
        assert self.state.tile_placed is not None
        if self.state.tile_placed.placed_pos != e.tile.pos:
            raise ValueError(f"You placed a meeple on an invalid tile - {e.tile.pos}")

        if self.state.tile_placed.rotation != e.tile.rotation:
            raise ValueError(
                f"You placed a meeple on a valid tile with an invalid/mismatched rotation - {e.tile.rotation}"
            )

        player = self.state.players[player_id]
        if player._get_available_meeple() is None:
            raise ValueError("You placed a meeple - You don't have one availble")

        if e.placed_on not in VALID_MEEPLE_PLACEMENTS:
            raise ValueError(
                f"You placed a meeple on a invalid structure - Your Strcuture {e.placed_on}"
            )

        if e.placed_on not in [MONASTARY_IDENTIFIER]:
            if self.state._get_claims(self.state.tile_placed, e.placed_on):
                raise ValueError(
                    "You tried placing a meeple on an unclaimable Structure - \
                    adjacent structure claimed by an opponent"
                )

            if (
                self.state.tile_placed.internal_edges[e.placed_on]
                not in VALID_STRUCTURE_CLAIMS
            ):
                raise ValueError(
                    f"You placed a meeple on a invalid edge - Edge Strcuture is {self.state.tile_placed.internal_edges[e.placed_on]}"
                )

        if e.placed_on == MONASTARY_IDENTIFIER:
            if TileModifier.MONASTARY not in self.state.tile_placed.modifiers:
                raise ValueError(
                    "You tried placing a meeple on a Monastary - \
                    There is no Monastary on the tile "
                )

    def _validate_place_meeple_pass(
        self, e: MovePlaceMeeplePass, query: BaseQuery, player_id: int
    ) -> None:
        pass
