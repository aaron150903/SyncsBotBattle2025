from helper.game import Game
from lib.interact.structure import StructureType
from lib.interact.tile import Tile
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
from typing import List

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

def get_valid_moves(game: Game):
    """
    Given Tile return all moves it can make 
    """
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    directions = {(0, -1): "top",(1, 0): "right",(0, 1): "bottom",(-1, 0): "left"}
    valid_moves = []

    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                for dx, dy in directions:
                    x1, y1 = x + dx, y + dy
                    if grid[y1][x1] is not None:
                        continue
                    for tile_index, tile in enumerate(game.state.my_tiles):
                        original_rotation = tile.rotation
                        for rot in range(4):
                            tile.rotation = rot
                            if game.can_place_tile_at(tile, x1, y1):
                                valid_moves.append((tile_index, tile._to_model(), x1, y1, rot))
                        tile.rotation = original_rotation  
    return valid_moves

def retrieve_details_of_structure(game: Game, structure: StructureType):
    """
    Pass in Structure Details and retrieve info on where they are located, amount of meeples in structure and size
    """
    pass

def compute_tile_score(game: Game, tile: Tile):
    """
    Use equation to compute the score of a tile
    """
    pass

def get_best_tile_placement(game: Game, tiles: List[Tile]):
    """
    Pass in all tiles we can validly place and return the best tile and placement based on equation computation
    """
    pass

def handle_place_tile(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
    pass

def handle_place_meeple(game: Game, bot_state: BotState, query: QueryPlaceMeeple) -> MovePlaceMeeple | MovePlaceMeeplePass:
    pass