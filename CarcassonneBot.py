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
import copy

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

def check_if_tile_completes_monastery(game: Game, tile: Tile, x: int, y: int):
    """
    Pass in Tile and coordinates it will check whether or not it completes monastery
    """
    pass           

def check_if_tile_completes_structure(tile: Tile, x: int, y: int, structure: StructureType, grid: list[list[Tile | None]]) -> bool:
    """
    Check if placing `tile` at (x, y) completes any structure (excluding monastery).

    Args:
        tile: Tile object with rotation set
        x, y: placement position
        grid: current board grid (2D list) of Tiles or None

    Returns:
        True if the placement closes any structure, else False.
    """
    # Make a shallow copy of grid and place tile at (x,y)
    new_grid = [row[:] for row in grid]
    new_grid[y][x] = tile

    def dfs(cx, cy, edge) -> bool:
        """
        Depth-first search to check if structure connected to tile at (cx, cy) is closed.
        Returns True if closed, False if open edge found.
        """

        stack = [(cx, cy, edge)]
        visited_local = set()

        while stack:
            tx, ty, tedge = stack.pop()
            if (tx, ty, tedge) in visited_local:
                continue
            visited_local.add((tx, ty, tedge))

            current_tile = new_grid[ty][tx]
            if current_tile is None:
                # Open edge: no tile where structure should connect
                return False

            # Check if current edge matches given structure type
            if current_tile.internal_edges[tedge] != structure:
                return False  # Edge not matching structure

            # Find opposite edge on neighbor tile
            opp_edge = Tile.get_opposite(tedge)
            # Neighbor tile coords:
            dx, dy = 0, 0
            if tedge == "top_edge":
                dy = -1
            elif tedge == "bottom_edge":
                dy = 1
            elif tedge == "left_edge":
                dx = -1
            elif tedge == "right_edge":
                dx = 1

            nx, ny = tx + dx, ty + dy
            if not (0 <= ny < len(new_grid) and 0 <= nx < len(new_grid[0])):
                # Off board: open edge
                return False

            neighbor_tile = new_grid[ny][nx]
            if neighbor_tile is None:
                return False  # open edge: structure not connected here

            # Check if neighbor tile edge matches structure 
            if neighbor_tile.internal_edges[opp_edge] != structure:
                return False  # neighbor edge not given structure -> open edge

            # Add all other structure edges of current tile except the one we came from
            for e in Tile.get_edges():
                if e == tedge:
                    continue
                if current_tile.internal_edges[e] == structure and (tx, ty, e) not in visited_local:
                    stack.append((tx, ty, e))

            # Add the corresponding edges from neighbor tile too
            for ne in Tile.get_edges():
                if ne == opp_edge:
                    continue
                if neighbor_tile.internal_edges[ne] == structure and (nx, ny, ne) not in visited_local:
                    stack.append((nx, ny, ne))

        return True

    # For every structure edge on the newly placed tile, check if the structure is closed starting there
    for edge in Tile.get_edges():
        if tile.internal_edges[edge] == structure:
            if not dfs(x, y, edge):
                return False

    # All structure edges checked, all closed
    return True


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
