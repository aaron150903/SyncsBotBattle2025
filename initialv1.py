from helper.game import Game
from lib.interact.tile import Tile, TileModifier
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

from collections import deque
from typing import Set, Tuple
from typing import Callable, Iterator
from copy import deepcopy
import math


class BotState:
    """A class for us to locally the state of the game and what we find relevant"""

    def __init__(self):
        self.last_tile: Tile | None = None
        self.meeples_placed: int = 0
        self.strat_pref = 'B'
        self.monastary_points = []
    
    def update_strat_pref(self, game_state):
        curr_points = game_state.points
        opponent_points = [
            p.points
            for p in game_state.players.values()
            if p.player_id != game_state.me.player_id
        ]

        if opponent_points and max(opponent_points) - curr_points > 10:
            self.strat_pref = 'A'

        elif max(opponent_points) == curr_points:
            #only need to be defensive if we're winning, otherwise choose balanced.
            self.strat_pref = 'D'

        else:
            self.strat_pref = 'B'

    def add_monastary(self,x1: int, y1: int):
        """
        Method will update points on grid where we have monastary (it will be centre point)
        """
        self.monastary_points.append((x1,y1))
        print(self.monastary_points, flush=True)

def main():
    game = Game()
    bot_state = BotState()

    while True:
        query = game.get_next_query()
        bot_state.update_strat_pref(game.state)
        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    print("placing tile")
                    return handle_place_tile(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    print("meeple")
                    return handle_place_meeple_advanced(game, bot_state, q)
                case _:
                    assert False

        print("sending move")
        game.send_move(choose_move(query))


def simulate_validate_place_tile(game, tile, rotation, x, y):
    """
    Simulate placement validation for tile before calling game.move_place_tile.
    Checks edge matching, river connection, U-turns, and adjacency.
    """

    grid = game.state.map._grid
    rotated_tile = deepcopy(tile)
    rotated_tile.rotate_clockwise(rotation)

    directions = {
        "top_edge": (0, -1),
        "right_edge": (1, 0),
        "bottom_edge": (0, 1),
        "left_edge": (-1, 0),
    }

    river_flag = False
    river_connections = 0
    has_any_neighbor = False

    for edge, (dx, dy) in directions.items():
        nx, ny = x + dx, y + dy
        if not (0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH):
            continue

        neighbor = grid[ny][nx]
        edge_structure = rotated_tile.internal_edges[edge]

        if edge_structure == StructureType.RIVER:
            river_flag = True

        if neighbor:
            has_any_neighbor = True
            neighbor_edge = Tile.get_opposite(edge)
            neighbor_structure = neighbor.internal_edges[neighbor_edge]

            if neighbor_structure != edge_structure:
                return False  # Edge mismatch

            if edge_structure == StructureType.RIVER:
                river_connections += 1
                if river_connections > 1:
                    return False  # River connects to more than 1 river

        elif edge_structure == StructureType.RIVER:
            # Direct U-turn check
            forecast_x = x + dx
            forecast_y = y + dy
            for ddx, ddy in directions.values():
                check_x = forecast_x + ddx
                check_y = forecast_y + ddy
                if (check_x != x or check_y != y) and (0 <= check_x < MAX_MAP_LENGTH and 0 <= check_y < MAX_MAP_LENGTH):
                    if grid[check_y][check_x] is not None:
                        return False

            # Indirect U-turn check (2 steps ahead)
            forecast_x = x + 2 * dx
            forecast_y = y + 2 * dy
            if 0 <= forecast_x < MAX_MAP_LENGTH and 0 <= forecast_y < MAX_MAP_LENGTH:
                if grid[forecast_y][forecast_x] is not None:
                    return False
                
    if not has_any_neighbor:
        return False  # Must be adjacent to something
    
    if river_flag and river_connections == 0:
        return False  # River tile not connected

    return True


def is_river_tile(tile) -> bool:
    """
    Checks if a tile contains any river edges.
    """
    return any(edge == StructureType.RIVER for edge in tile.internal_edges.values())


def analyze_structure_from_edge(
    start_tile: "Tile",
    start_edge: str,
    grid,
) -> tuple["StructureType", Set[tuple["Tile", str]], bool, Set["Tile"]]:
    """
    Method is based on logic in transverse connected component and method will take in edge return the 
    structure, all tiles in the structure and whether its complete or not.
    """
    # Track all visited tiles
    visited: Set[tuple["Tile", str]] = set()
    queue = deque([(start_tile, start_edge)])

    # Tile has no structure
    if start_edge not in start_tile.internal_edges:
        return None, set(), True, 0  
    
    # Get the structure we will be looking at as each tile has 4 edges with potential structure
    structure_type = start_tile.internal_edges[start_edge]
    structure_bridge = TileModifier.get_bridge_modifier(structure_type)

    # Have a set for getting the connected components and one for unique tiles in structure so we can capture size
    connected_parts: Set[tuple["Tile", str]] = set()
    unique_tiles: Set["Tile"] = set()
    is_complete = True

    # Begin BFS
    while queue:
        tile, edge = queue.popleft()
        if (tile, edge) in visited:
            continue

        # Update visited as we already went through it
        visited.add((tile, edge))
        connected_parts.add((tile, edge))
        unique_tiles.add(tile)

        internal_edges = {edge}

        # Find all same-type internal edges on this tile which are the same as structure
        for adj_edge in Tile.adjacent_edges(edge):
            if tile.internal_edges.get(adj_edge) == structure_type:
                if not (TileModifier.BROKEN_CITY in tile.modifiers and structure_type == StructureType.CITY):
                    internal_edges.add(adj_edge)

        # Handle bridge through center if applicable
        if (len(internal_edges) == 1 and structure_bridge and structure_bridge in tile.modifiers):
            opposite = Tile.get_opposite(edge)
            if StructureType.is_compatible(structure_type,tile.internal_edges.get(opposite)):
                internal_edges.add(opposite)

        # Traverse neighbors
        for internal_edge in internal_edges:
            if tile.placed_pos is None:
                is_complete = False
                continue

            # Get the tile which is located based on edge and opposite edge
            neighbor = Tile.get_external_tile(internal_edge, tile.placed_pos, grid)
            opposite_edge = Tile.get_opposite(internal_edge)

            # If there is no tile on the left then return false
            if not neighbor:
                is_complete = False
                continue

            neighbor_type = neighbor.internal_edges.get(opposite_edge)

            # Special ROAD_START handling
            if structure_type == StructureType.ROAD and neighbor_type == StructureType.ROAD_START:
                continue

            # If neighbour is not same type as structure type then return false
            if neighbor_type != structure_type:
                is_complete = False
                continue

            if (neighbor, opposite_edge) not in visited:
                queue.append((neighbor, opposite_edge))

    return structure_type, connected_parts, is_complete, unique_tiles

def get_shield_cities_count(tiles) -> int:
    """
    Given a collection of tiles finds how many are shielded
    """
    count = 0
    for tile in tiles:
        if TileModifier.EMBLEM in tile.modifiers:
            count+=1
    return count

def count_existing_neighbours_monastery(game, move):
    curr_x, curr_y = move['tx'], move['ty']
    curr_grid = game.state.map._grid
    total_neighbours = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = curr_x + dx, curr_y + dy
            if not (0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH):
                continue
            if curr_grid[ny][nx] is not None:
                total_neighbours += 1
    return total_neighbours

def city_extension_bonus(game, connected_parts):
    claimed_by_us = False
    #connected parts is (tile, edge)
    claim_bonus = 3
    for tile, edge in connected_parts:
        claims = game.state._get_claims(tile, edge)
        for claim in claims:
            if claim == game.state.me.player_id:
                claimed_by_us = True
                break
    if claimed_by_us:
        return claim_bonus
    return 0

def evaluate_move(game, move, bot_state: BotState):
    score = 0
    tile = move["tile"]
    # Might need a different method for monastery checking this is just seeing if tile itself is monastery (we could check surronding 3x3 tiles)
    if TileModifier.MONASTARY in tile.modifiers:
        neighbour_weighting = count_existing_neighbours_monastery(game, move)*1.1
        score+= (4.5 + neighbour_weighting) 
    else:
        # Otherwise see if tile is surronding a monastary we own and work on finishing off this monastary
        for monastary_point in bot_state.monastary_points:
            # This means the tile will surrond the monastary
            empty = 0
            if abs(move["tx"]-monastary_point[0]) <= 1 and abs(move["ty"]-monastary_point[1]) <= 1:
                # Now determine how many tiles are empty around this monastary
                for horizontal in range(-1,2):
                    for vertical in range(-1,2):
                        if game.state.map._grid[monastary_point[1]+vertical][monastary_point[0]+horizontal] is None:
                            empty += 1
                print(f'Monastary we own has {empty} empty tiles',flush=True)
                # Now determine how many points we want to add based on this subtract 9 from the amount of tiles needed
                score += 9 - empty 
                break
        
    x, y = move["tx"], move["ty"]
    # Simulate a new grid 
    new_grid = [row[:] for row in game.state.map._grid]
    tile.placed_pos = (x, y)
    new_grid[y][x] = tile

    # Iterate through each edge in tile to see what structure it completes if any
    edges = tile.get_external_tiles(new_grid).keys()
    for edge in edges:
        structure_type, connected_parts, is_complete, unique_tiles = analyze_structure_from_edge(tile,edge,new_grid)
        print(structure_type,connected_parts,is_complete,unique_tiles,flush=True)
        # Get the size of the structure
        structure_size = len(unique_tiles)
        # Now see the type of structure and if it is a city add more points
        if structure_type == StructureType.CITY:
            claim_bonus = city_extension_bonus(game, connected_parts=connected_parts)
            score += claim_bonus
            if is_complete:
                score += structure_size * 2 + get_shield_cities_count(unique_tiles) * 2
                print(f'This many shielded tiles found: {get_shield_cities_count(unique_tiles)}', flush=True)
            else:
                score += structure_size + get_shield_cities_count(unique_tiles) * 2
                print(f'This many shielded tiles found: {get_shield_cities_count(unique_tiles)}', flush=True)
        elif structure_type == StructureType.ROAD or structure_type == StructureType.ROAD_START:
            if is_complete:
                score += structure_size
            else:
                pass
    print(score,flush=True)
    return score

def get_best_evaluated_move(game: Game, legal_moves: set, bot_state: BotState):
    best_move = list(legal_moves)[0]
    best_score = 0
    for move in legal_moves:
        score = evaluate_move(game, move, bot_state)
        if score>best_score:
            best_score = score
            best_move = move
    return best_move


def handle_place_tile(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
    """
    Handles placing a tile from hand by checking river phase and attempting valid moves.
    Uses simulated validation to avoid engine rejections.
    """

    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    latest_tile = game.state.map.placed_tiles[-1]
    latest_pos = latest_tile.placed_pos
    assert latest_pos

    directions = {(1, 0): "left_edge", (0, 1): "top_edge", (-1, 0): "right_edge", (0, -1): "bottom_edge"}

    # Check if we are in river stage:
    is_river = False
    for tile_index, tile in enumerate(game.state.my_tiles):
        if is_river_tile(tile):
            is_river = True
            break

    # Determine valid locations to potentially place tiles
    candidates = set()
    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < width and 0 <= ny < height and grid[ny][nx] is None):
                        candidates.add((nx, ny))
    legal_moves = []

    for tile_index, tile in enumerate(game.state.my_tiles):
        if is_river:
            print("In river stage", flush=True)
            for (dx, dy) in directions.keys():
                tx, ty = latest_pos[0] + dx, latest_pos[1] + dy
                for rotation in range(4):
                    if simulate_validate_place_tile(game, tile, rotation, tx, ty):
                        tile.rotate_clockwise(rotation)
                        tile.placed_pos = (tx, ty)
                        bot_state.last_tile = tile
                        return game.move_place_tile(query, tile._to_model(), tile_index)
                tile.rotate_clockwise(4)  # Reset tile
        else:
            print("In normal stage", flush=True)
            for (x, y) in candidates:
                for rotation in range(4):
                    if simulate_validate_place_tile(game, tile, rotation, x, y):
                        legal_moves.append({"tile": tile,"rotation": rotation,"tx": x,"ty": y, "index":tile_index})

            if len(legal_moves) == 0:
                return brute_force_tile(game,bot_state,query)
            
            move = get_best_evaluated_move(game, legal_moves, bot_state)
            # If tile is monastary add its positions to our bot state
            if TileModifier.MONASTARY in move["tile"].modifiers:
                bot_state.add_monastary(move["tx"],move["ty"])
            move["tile"].rotate_clockwise(move["rotation"])
            move["tile"].placed_pos = (move["tx"], move["ty"])
            bot_state.last_tile = move["tile"]
            return game.move_place_tile(query, move["tile"]._to_model(), move["index"])

def brute_force_tile(
    game: Game, bot_state: BotState, query: QueryPlaceTile
) -> MovePlaceTile:
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    directions = {
        (0, 1): "top",
        (1, 0): "right",
        (0, -1): "bottom",
        (-1, 0): "left",
    }
    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                print(f"Checking if tile can be placed near tile - {grid[y][x]}")
                for tile_index, tile in enumerate(game.state.my_tiles):
                    for direction in directions:
                        dx, dy = direction
                        x1, y1 = (x + dx, y + dy)
                        for rotation in range(4):
                            if simulate_validate_place_tile(game, tile, rotation, x1, y1):
                                tile.rotate_clockwise(rotation)
                                bot_state.last_tile = tile
                                bot_state.last_tile.placed_pos = (x1, y1)
                                # If tile is monastary add its positions to our bot state
                                if TileModifier.MONASTARY in tile.modifiers:
                                    bot_state.add_monastary(x1,y1)
                                return game.move_place_tile(query, tile._to_model(), tile_index)
                        tile.rotate_clockwise(4)  # Reset tile to original orientation

def evaluate_meeple_placement(structure_type: StructureType, bot_state: BotState):
    structure_points = {StructureType.CITY: 3, StructureType.MONASTARY: 3, StructureType.ROAD: 2, StructureType.ROAD_START: 2, StructureType.GRASS: 1, StructureType.RIVER: 0}
    strategy_bonus = 2 if bot_state.strat_pref == 'A' else 0
    total_points = (structure_points[structure_type] if structure_type in structure_points else 0) + strategy_bonus
    return (total_points)

def handle_place_meeple_advanced(game: Game, bot_state: BotState, query: QueryPlaceMeeple):
    if bot_state.last_tile is None or bot_state.meeples_placed == 7:
        return game.move_place_meeple_pass(query)
        
    
    structures = game.state.get_placeable_structures(bot_state.last_tile._to_model())
    tile_model = bot_state.last_tile
    bot_state.last_tile = None

    print(f"Available structures: {structures}")  # Debug print
    
    if structures:
        structure_scores = []
        
        x, y = tile_model.placed_pos
        placed_tile = game.state.map._grid[y][x]
        
        for edge, structure in structures.items():
            # Use the placed tile from the map for claims checking
            is_claimed = game.state._get_claims(placed_tile, edge)
            is_completed = game.state._check_completed_component(placed_tile, edge)
            
            print(f"Edge: {edge}, Structure: {structure}, Claimed: {is_claimed}, Completed: {is_completed}")
            #Monasteries can't be claimed or completed prior to being handed out, the person drawing the monastary
            #will be the first to claim or complete it
            if edge != 'MONASTARY' and (is_claimed or is_completed):
                print(f"Skipping {edge} - claimed: {is_claimed}, completed: {is_completed}")
                continue
            else:
                score = evaluate_meeple_placement(structure, bot_state)
                structure_scores.append((score, edge, structure))
                print(f"Added to consideration: Edge={edge}, Structure={structure}, Score={score}")
        
        print(f"Final structure_scores: {structure_scores}")
        
        if not structure_scores:
            print("No valid structures found, passing")
            return game.move_place_meeple_pass(query)
            
        best_meeple_placement = max(structure_scores, key=lambda x: x[0])
        best_score, best_edge, best_structure = best_meeple_placement[0], best_meeple_placement[1], best_meeple_placement[2]
       
        print(f"Best placement: Edge={best_edge}, Structure={best_structure}, Score={best_score}")
        
        if (bot_state.strat_pref == 'A' and best_score > 1):
            bot_state.meeples_placed += 1
            return game.move_place_meeple(query, tile_model._to_model(), placed_on=best_edge)
        elif best_score > 2:
            bot_state.meeples_placed += 1
            return game.move_place_meeple(query, tile_model._to_model(), placed_on=best_edge)
        else:
            print(f"Score {best_score} not high enough, passing")
            return game.move_place_meeple_pass(query)
            
    return game.move_place_meeple_pass(query)

if __name__ == "__main__":
    main()