"""
Demo bot.

Blame: Maurice

Logic:
- Always place tiles next to the last placed tile. If not possible,
- Greedily place meeple in monastary -> next valid thing


Flaws:
- Does not take advantage of rotating the tiles
- Does not care about moves of other players other than for the sake of validation
- Does not account for other existing structures/linking other structures
- As of writing, cannot pass river stage

"""

from helper.game import Game
from lib.interact.tile import Tile
from lib.interface.events.moves.move_place_tile import MovePlaceTile
from lib.interface.events.moves.move_place_meeple import (
    MovePlaceMeeple,
    MovePlaceMeeplePass,
)
from lib.interface.queries.typing import QueryType
from lib.interface.queries.query_place_tile import QueryPlaceTile
from lib.interface.queries.query_place_meeple import QueryPlaceMeeple
from lib.interface.events.moves.typing import MoveType
from lib.config.map_config import MAX_MAP_LENGTH
from lib.config.map_config import MONASTARY_IDENTIFIER
from lib.interact.structure import StructureType


"""A class for us to locally the state of the game and what we find relevant"""


class BotState:
    def __init__(self):
        self.last_tile: Tile | None = None


def main():
    game = Game()
    bot_state = BotState()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    print("placing tile")
                    return handle_place_tile(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    print("meeple")
                    return handle_place_meeple(game, bot_state, q)
                case _:
                    assert False

        print("sending move")
        game.send_move(choose_move(query))


def handle_place_tile(
    game: Game, bot_state: BotState, query: QueryPlaceTile
) -> MovePlaceTile:
    """
    Find the most recently placed tile and try to place a new tile adjacent to it.
    Tries directions in order: right, bottom, left, top
    """
    grid = game.state.map._grid

    # The direction of placing the tile in reference to the last placed tile
    directions = {
        (1, 0): "left_edge",  # right
        (0, 1): "top_edge",  # bottom
        (-1, 0): "right_edge",  # left
        (0, -1): "bottom_edge",  # top
    }
    # Will either be the latest tile
    latest_tile = game.state.map.placed_tiles[-1]
    latest_pos = latest_tile.placed_pos

    print(game.state.my_tiles)
    assert latest_pos

    # Try to place a tile adjacent to the latest tile
    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
        river_flag = False
        for find_edge in directions.values():
            if tile_in_hand.internal_edges[find_edge] == StructureType.RIVER:
                river_flag = True
                break
        for (dx, dy), edge in directions.items():
            target_x = latest_pos[0] + dx
            target_y = latest_pos[1] + dy

            # Check bounds
            if not (0 <= target_x < MAX_MAP_LENGTH and 0 <= target_y < MAX_MAP_LENGTH):
                continue

            # Check if position is empty
            if grid[target_y][target_x] is not None:
                continue

            # Cheeck if this is a river tile
            # Try placing the tile at this position

            if game.can_place_tile_at(tile_in_hand, target_x, target_y):
                if (
                    river_flag
                    and tile_in_hand.internal_edges[edge] != StructureType.RIVER
                ):
                    tile_in_hand.rotate_clockwise(1)
                    continue
                else:
                    bot_state.last_tile = tile_in_hand
                    bot_state.last_tile.placed_pos = (target_x, target_y)
                    print(
                        bot_state.last_tile.placed_pos,
                        tile_hand_index,
                        tile_in_hand.rotation,
                        tile_in_hand.tile_type,
                        flush=True,
                    )
                    return game.move_place_tile(
                        query, tile_in_hand._to_model(), tile_hand_index
                    )


def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeplePass | MovePlaceMeeple:
    """
    Try to place a meeple on the most recently placed tile.
    Priority order: monastery -> Anything else
    """
    recent_tile = bot_state.last_tile
    if not recent_tile:
        return game.move_place_meeple_pass(query)

    # Priority order for meeple placement
    placement_priorities = [
        MONASTARY_IDENTIFIER,  # monastery
        "top_edge",
        "right_edge",
        "bottom_edge",
        "left_edge",  # edges
    ]

    for edge in placement_priorities:
        # Check if this edge has a valid structure and is unclaimed
        if edge == MONASTARY_IDENTIFIER:
            # Check if tile has monastery and it's unclaimed
            if (
                hasattr(recent_tile, "modifiers")
                and any(mod.name == "MONESTERY" for mod in recent_tile.modifiers)
                and recent_tile.internal_claims.get(MONASTARY_IDENTIFIER) is None
            ):
                return game.move_place_meeple(
                    query, recent_tile._to_model(), MONASTARY_IDENTIFIER
                )
        else:
            # Check if edge has a claimable structure
            structure = recent_tile.internal_edges.get(edge)
            if (
                structure
                and structure != StructureType.GRASS
                and recent_tile.internal_claims.get(edge) is None
            ):
                # Check if the structure is actually unclaimed (not connected to claimed structures)
                if not game.state._get_claims(recent_tile, edge):
                    return game.move_place_meeple(query, recent_tile._to_model(), edge)

    # No valid placement found, pass
    return game.move_place_meeple_pass(query)


# """Copied from simple.py: brute force a valid tile"""


# def brute_force_tile(
#     game: Game, bot_state: BotState, query: QueryPlaceTile
# ) -> MovePlaceTile:
#     grid = game.state.map._grid
#     height = len(grid)
#     width = len(grid[0]) if height > 0 else 0

#     directions = {
#         (0, 1): "top",
#         (1, 0): "right",
#         (0, -1): "bottom",
#         (-1, 0): "left",
#     }

#     print(game.state.event_history)

#     print("Cards", game.state.my_tiles)

#     for y in range(height):
#         for x in range(width):
#             if grid[y][x] is not None:
#                 print(f"Checking if tile can be placed near tile - {grid[y][x]}")
#                 for tile_index, tile in enumerate(game.state.my_tiles):
#                     for direction in directions:
#                         dx, dy = direction
#                         x1, y1 = (x + dx, y + dy)

#                         if game.can_place_tile_at(tile, x1, y1):
#                             bot_state.last_tile = tile._to_model()
#                             bot_state.last_tile.pos = (x1, y1)
#                             return game.move_place_tile(query, tile, tile_index)


if __name__ == "__main__":
    main()
