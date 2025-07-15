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
from lib.interact.tile import Tile
from lib.interact.tile import create_base_tiles
from typing import List
import copy


remaining_tiles = create_base_tiles()

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

def calculate_probability(top: StructureType, right: StructureType, left: StructureType, bottom: StructureType):
    matching_tile_amount = 0
    for tile in remaining_tiles:
        for rotation in range(1,5):
            tile.rotate_clockwise(rotation)
            edges = tile.internal_edges
            if ((top==None or edges["top_edge"]==top) and (right==None or edges["right_edge"]==right) and (left==None or edges["left_edge"]==left) and (bottom==None or edges["bottom_edge"]==bottom)):
                matching_tile_amount+=1
                break
    return (matching_tile_amount / remaining_tiles)

def update_tile_distribution(used_tile: Tile):
    id = used_tile.tile_type
    tile = None
    for t in remaining_tiles:
        if t.tile_type==id:
            tile = t
            break
    remaining_tiles.remove(tile)

def get_value_of_completed_structure(tile: Tile, structure_type: StructureType, x, y):
    '''Given a tile and the type of structure it completes, return the total value of that structure
    '''
    return 1

def get_structure_claimed(game: Game, tile: Tile, structure_type: StructureType, x: int, y: int, rot: int):
    '''Returns if we already own this structure or not. Even if we don't have a meeple on this current tile
    we may still already own the structure. Now that theres no stealing based on number of meeples in the structure, 
    we don't need to add more meeples on a structure we already own.'''
    return True

def get_value_of_incomplete_structure(game: Game, tile: Tile, structure_type: StructureType, x: int, y: int):
    '''Returns the potential value of an incomplete structure by finding the current value of the structure if scoring
    was to be done now and adds some extra benefit which we'd get if we completed this structure and had a meeple on it. This will be scaled
    by the probability of completing the structure'''
    return 5

def compute_base_score(game: Game, tile: Tile, x: int, y: int, rot: int):
    '''Gets the score of this tile if we don't put a meeple on it'''
    total_base_score = 0
    possible_completed_structures = [tile.top_edge, tile.right_edge, tile.bottom_edge, tile.left_edge]
    for possible_completed_structure in possible_completed_structures:
        if (get_structure_claimed(tile, possible_completed_structure, x, y)) and check_if_tile_completes_structure(tile, x, y, possible_completed_structure, game.state.map._grid): 
            total_base_score += get_value_of_completed_structure(tile, possible_completed_structure, x, y)
    return total_base_score

def compute_incremental_score(game: Game, tile: Tile, x: int, y: int, rot: int):
    '''Gets the additional value of this tile when you place a meeple on one of the tile's structures.
    Returns the score you get by placing a meeple on the most valuable asset on the tile which is determined
    by the value of the asset and probability of completing that asset. Also returns where to place the meeple.
    '''
    possible_meeple_placements = [tile.top_edge, tile.right_edge, tile.bottom_edge, tile.left_edge]
    best_meeple_placement_score = 0
    best_meeple_placement = None
    for meeple_placement in possible_meeple_placements:
        if (get_structure_claimed(game, tile, meeple_placement, x, y)): continue
        #we'll probably need to change how we are calculating the probabilities. 
        expected_benefits = (get_value_of_incomplete_structure(game, tile, meeple_placement, x, y) * calculate_probability())
        if expected_benefits > best_meeple_placement_score:
            best_meeple_placement = meeple_placement
            best_meeple_placement_score = expected_benefits
    return (best_meeple_placement_score, best_meeple_placement)


def compute_tile_score(game: Game, tile: Tile, x: int, y: int, rot: int):
    '''Returns total score of a tile'''
    base_score = compute_base_score(game, tile, x, y, rot)
    incremental_score_tuple = compute_incremental_score(game, tile, x, y, rot)
    incremental_score, meeple_placement = incremental_score_tuple[0], incremental_score_tuple[1]
    if incremental_score <= 0:
        #theres no gain from adding a meeple
        meeple_placement = None
    return (base_score+incremental_score, meeple_placement)


def get_best_tile_placement(game: Game, tiles: List[Tile]):
    """
    Pass in all tiles we can validly place and return the best tile and placement based on equation computation
    """
    valid_placements = get_valid_moves(game)
    best_tile_placement = None
    best_tile_placement_score = float('-inf')
    best_meeple_placement = None
    for valid_placement in valid_placements:
        curr_placement_tuple = compute_tile_score(game, valid_placement[1], valid_placement[2], valid_placement[3], valid_placement[4])
        tile_placement_score, meeple_placement = curr_placement_tuple[0], curr_placement_tuple[1]
        if tile_placement_score > best_tile_placement_score:
            best_tile_placement_score = tile_placement_score
            best_meeple_placement = meeple_placement
            best_tile_placement = valid_placement[1]
    return (best_tile_placement, best_meeple_placement)

def handle_place_tile(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
    pass

def handle_place_meeple(game: Game, bot_state: BotState, query: QueryPlaceMeeple) -> MovePlaceMeeple | MovePlaceMeeplePass:
    pass
