#### DISCLAIMER ####
# This was produced with the assistance of GPT 4o

from helper.game import Game
from lib.interface.events.moves.move_place_meeple import (
    MovePlaceMeeple,
    MovePlaceMeeplePass,
)
from lib.interface.events.moves.move_place_tile import MovePlaceTile
from lib.interface.queries.typing import QueryType
from lib.interface.queries.query_place_tile import QueryPlaceTile
from lib.interface.queries.query_place_meeple import QueryPlaceMeeple
from lib.interface.events.moves.typing import MoveType
from lib.models.tile_model import TileModel


class BotState:
    def __init__(self) -> None:
        self.last_tile: TileModel | None = None


def main() -> None:
    game = Game()
    bot_state = BotState()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    return handle_place_tile(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    return handle_place_meeple(game, bot_state, q)

        game.send_move(choose_move(query))


def handle_place_tile(
    game: Game, bot_state: BotState, query: QueryPlaceTile
) -> MovePlaceTile:
    """Place a tile in the first valid location found on the board - brute force"""
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    directions = {
        (0, -1): "top",
        (1, 0): "right",
        (0, 1): "bottom",
        (-1, 0): "left",
    }

    print(game.state.event_history)

    print("Tiles", game.state.my_tiles)

    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                for tile_index, tile in enumerate(game.state.my_tiles):
                    print(
                        f"Checking if tile {tile.tile_type} can be placed near tile - {grid[y][x]}, at {x, y}"
                    )
                    for direction in directions:
                        dx, dy = direction
                        x1, y1 = (x + dx, y + dy)

                        if game.can_place_tile_at(tile, x1, y1):
                            tile.placed_pos = (x1, y1)
                            bot_state.last_tile = tile._to_model()
                            print("", end="", flush=True)

                            return game.move_place_tile(
                                query, tile._to_model(), tile_index
                            )

    raise ValueError(
        "No valid tile placement found - this feature has not been implmented yet"
    )


def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """Try to place a meeple on the tile just placed, or pass."""
    assert bot_state.last_tile is not None
    structures = game.state.get_placeable_structures(bot_state.last_tile)

    x, y = bot_state.last_tile.pos
    tile = game.state.map._grid[y][x]

    assert tile is not None

    tile_model = bot_state.last_tile
    bot_state.last_tile = None

    if structures:
        for edge, _ in structures.items():
            if game.state._get_claims(tile, edge):
                continue

            else:
                return game.move_place_meeple(query, tile_model, placed_on=edge)

    return game.move_place_meeple_pass(query)


if __name__ == "__main__":
    main()
