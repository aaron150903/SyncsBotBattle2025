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

class TileEncoding:
    def __init__(self, top: str, right: str, bottom: str, left: str, center: str):
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left
        self.center = center
        self.rotation = 0

    def rotate_clockwise(self, num: int) -> None:
        for i in range(num):
            (
                self.right,
                self.bottom,
                self.left,
                self.top,
            ) = (
                self.top,
                self.right,
                self.bottom,
                self.left,
            )
        self.rotation += num
        self.rotation %= 4

    def __eq__(self, other) -> bool:
        for i in range(1,5):
            self.rotate_clockwise(i)
            if self.top==other.top and self.right==other.right and self.bottom==other.bottom and self.left==other.left:
                return True
        return False

class TileProbability:
    def __init__(self):
        # F = Field, R = Road, M = Monastery, C = City
        # I've put the features of a tile based on how they appear if you go clockwise around the tile starting from the top left
        self.tile_map = {
            # A: Monastery with road (2x)
            TileEncoding('F', 'F', 'R', 'F', 'M'): 2,
            
            # B: Monastery alone (4x)
            TileEncoding('F', 'F', 'F', 'F', 'M'): 4,
            
            # C: Large city with pennant (1x)
            TileEncoding('C', 'C', 'C', 'C', 'C'): 1,
            
            # D: City corner with road (4x) - includes start tile
            TileEncoding('F', 'R', 'F', 'C', 'F'): 4,
            
            # E: City corner (5x)
            TileEncoding('F', 'F', 'F', 'C', 'F'): 5,
            
            # F: City on two adjacent sides (2x)
            TileEncoding('C', 'C', 'F', 'F', 'F'): 2,
            
            # G: City on opposite sides (1x)
            TileEncoding('C', 'F', 'C', 'F', 'C'): 1,
            
            # H: City on three sides (3x)
            TileEncoding('C', 'C', 'F', 'C', 'C'): 3,
            
            # I: City on three sides with road (2x)
            TileEncoding('C', 'C', 'R', 'C', 'C'): 2,
            
            # J: City corner with road on adjacent side (3x)
            TileEncoding('C', 'R', 'F', 'C', 'F'): 3,
            
            # K: City corner with road on opposite side (3x)
            TileEncoding('C', 'F', 'R', 'C', 'F'): 3,
            
            # L: City corner with road on far side (3x)
            TileEncoding('C', 'F', 'F', 'R', 'F'): 3,
            
            # M: City with pennant on diagonal (2x)
            TileEncoding('C', 'F', 'F', 'C', 'F'): 2,
            
            # N: City with pennant on three sides (3x)
            TileEncoding('C', 'C', 'F', 'C', 'C'): 3,
            
            # O: City with pennant on diagonal plus road (2x)
            TileEncoding('C', 'R', 'F', 'C', 'F'): 2,
            
            # P: City with pennant on three sides plus road (3x)
            TileEncoding('C', 'C', 'R', 'C', 'C'): 3,
            
            # Q: City with pennant on two adjacent sides (1x)
            TileEncoding('C', 'C', 'F', 'F', 'C'): 1,
            
            # R: City with pennant on two adjacent sides (3x)
            TileEncoding('C', 'C', 'F', 'F', 'C'): 3,
            
            # S: City with pennant on two adjacent sides plus road (2x)
            TileEncoding('C', 'C', 'R', 'F', 'C'): 2,
            
            # T: City with pennant on two adjacent sides plus road (1x)
            TileEncoding('C', 'C', 'R', 'F', 'C'): 1,
            
            # U: Straight road (8x)
            TileEncoding('R', 'F', 'R', 'F', 'F'): 8,
            
            # V: Curved road (9x)
            TileEncoding('R', 'R', 'F', 'F', 'F'): 9,
            
            # W: T-junction road (4x)
            TileEncoding('R', 'R', 'F', 'R', 'F'): 4,
    
            # X: 4-way intersection (1x)
            TileEncoding('R', 'R', 'R', 'R', 'F'): 1,
        }
        self.total_tiles = 72

    def calculate_probability(self, required_tile: TileEncoding):
        matching_tile_amount = 0
        for tile in self.tile_map:
            matching_tile_amount = self.tile_map[tile] if (tile.top_edge == required_tile.top_edge and tile.right_edge == required_tile.right_edge and tile.bottom_edge == required_tile.bottom_edge and tile.left_edge == required_tile.left_edge and tile.center == required_tile.center) else 0
        return (matching_tile_amount / self.total_tiles)
    
    def update_tile_distribution(self, used_tile: TileEncoding):
        matching_tile = None
        for tile in self.tile_map:
            matching_tile = tile if (used_tile.top_edge == matching_tile.top_edge and used_tile.right_edge == tile.right_edge and used_tile.bottom_edge == tile.bottom_edge and used_tile.left_edge == tile.bottom_edge and used_tile.center == tile.center) else None
        self.tile_map[matching_tile] = self.tile_map[matching_tile] - 1 if matching_tile != None else self.tile_map[matching_tile]


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

def check_if_tile_completes_monastery(Monastery_x: int, Monastery_y: int,x: int, y: int, grid: list[list[Tile | None]]):
    """
    Pass in Coordinates of Centre of Monastery 
    and Tile location which is vacant where we want to place tile
    and it will return whether or not it will be the final tile
    """
    for i in range(-1,2):
        for j in range(-1,2):
            # If we find a tile which is empty around monastery which is not vacant tile location we also found means more than 1 vacant
            if grid[Monastery_x+i][Monastery_y+j] is None and x != Monastery_x+i and y != Monastery_y+j:
                return False
    return True         

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
    Return a dictionary with:
    - locations: list of (x, y) where structure type appears
    - meeples: count of meeples on that structure type
    - size: number of tiles containing that structure type
    """
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0])
    
    structure_locations = []
    meeple_count = 0
    tile_count = 0

    for y in range(height):
        for x in range(width):
            tile = grid[y][x]
            if tile is None:
                continue

            if structure in tile.internal_edges.values():
                structure_locations.append((x, y))
                tile_count += 1

                if hasattr(tile, "meeples"):
                    for meeple in tile.meeples:
                        if meeple.structure_type == structure:
                            meeple_count += 1

    return {
        "locations": structure_locations,
        "meeples": meeple_count,
        "size": tile_count,
    }



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
