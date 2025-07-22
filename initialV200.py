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
        self.move = 0
        # Store the structures we have claimed will track all
        self.claimed_structures = []
        self.unclaimed_open_spots = {}
    
    def update_strat_pref(self, game_state):
        curr_points = game_state.me.points
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

    def obtain_max_points(self, game_state):
        points = [p.points for p in game_state.players.values()]
        return max(points[0],points[1],points[2],points[3])
    
    def is_winner(self, game_state):
        values = [v for v in game_state.players.values()]
        sorted(values, key = lambda x: x.points)
        return values[-1].player_id==game_state.me.player_id

    def add_monastary(self,x1: int, y1: int):
        """
        Method will update points on grid where we have monastary (it will be centre point)
        """
        self.monastary_points.append((x1,y1))
        print(self.monastary_points, flush=True)

    def add_claimed_structure(self,structure_type: StructureType, tile, edge):
        """
        Method will help us keep track of our structures method for adding structures other than monastary
        """
        self.claimed_structures.append((structure_type,tile,edge))

    def updated_claimed_structures(self, game):
        """
        Method will update claimed structures if a given structure has been completed
        """
        new_set = []
        for structure in self.claimed_structures:
            if game.state._check_completed_component(structure[1], structure[2]):
                continue
            else:
                new_set.append(structure)
        # Update the structures we are tracking
        self.claimed_structures = new_set
                

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
        return None, set(), True, set()
    
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
    claim_bonus = 3
    penalty = -2  # For opponent claims
    bonus = 0
    has_our_claim = False
    has_enemy_claim = False
    
    for tile, edge in connected_parts:
        claims = game.state._get_claims(tile, edge)
        for claim in claims:
            if claim == game.state.me.player_id:
                has_our_claim = True
            else:
                has_enemy_claim = True

    if has_our_claim:
        bonus += claim_bonus
    if has_enemy_claim:
        bonus += penalty  # Apply penalty if any part is claimed by opponent
    return bonus

def opponent_monastary_extension_penalty(game,bot_state,move):
    """
    Given tile location sees if we are contributing to an opponent monastary
    """
    x, y = move['tx'], move['ty']
    grid = game.state.map._grid
    placement_around_opponent_monastary = False
    location = (-1,-1)
    for dx in range(-1,2):
        for dy in range(-1,2):
            # If we have found a tile which is monastary and it is not in our owned monastary we have found an enemy monastary
            if grid[y+dy][x+dx] is not None and TileModifier.MONASTARY in grid[y+dy][x+dx].modifiers and (x+dx,y+dy) not in bot_state.monastary_points:
                placement_around_opponent_monastary = True
                location = (x+dx,y+dy)
                print('Found enemy monastary!')
    
    if placement_around_opponent_monastary:
        # If there is an enemy monastary see how many tiles adjacent to it already
        placed_tiles = 0
        x, y = location
        for dx in range(-1,2):
            for dy in range(-1,2):
                if grid[y+dy][x+dx] is not None:
                    placed_tiles += 1
        print(f'Found opponent monastary with {placed_tiles} around it', flush=True)
        # Penalty will also consider how close we are to freeing opponent meeple
        return 2 * (1+(placed_tiles/9))
    # No penalty if we are not freeing anything
    return 0

def large_unclaimed_structures(game, size_threshold=3):
    grid = game.state.map._grid
    seen = set() #add all seen tile edge combos here so we don't explore components we've already seen
    result = {} #only add large enough unclaimed structures into here
    directions = {(0,-1):"top_edge",(1,0):"right_edge",(0,1):"bottom_edge",(-1,0):"left_edge"}

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            tile = grid[y][x]
            if not tile:
                continue
            for edge in directions.values():
                if (tile, edge) in seen:
                    #this edge was part of a component we've already seen
                    continue
                #tuple["StructureType", Set[tuple["Tile", str]], bool, Set["Tile"]]
                structure_type, comp_parts, is_complete, tiles = analyze_structure_from_edge(tile, edge, grid)
                if structure_type not in [StructureType.CITY, StructureType.ROAD, StructureType.ROAD_START]:
                    continue
                for (t1, e1) in comp_parts:
                    seen.add((t1, e1))
                structure_length = len(tiles)
                if structure_length > 5:
                    print(f"Unclaimed structure bonus {structure_length}")
                    print(f"Inspecting tile {tile.tile_type} at {(x,y)}")
                if is_complete or len(tiles) < size_threshold:
                    continue

                is_claimed = False
                for (t2, e2) in comp_parts:
                    num_claims = game.state._get_claims(t2, e2)
                    if len(num_claims) != 0:
                        is_claimed = True
                        break
                
                if is_claimed:
                    continue
                open_spots = forecast_number_of_tiles_needed_to_complete(structure_type, tiles, grid)
                result[(structure_type, frozenset(tiles))] = open_spots
    return result
                    
def add_bonus_for_unclaimed_structures(bot_state, move):
    x, y = move['tx'], move['ty']
    bonus = 0
    for (struct_type, struct_tiles), open_spots in bot_state.unclaimed_open_spots.items():
        if (x, y) in open_spots:
            size = len(struct_tiles)
            if struct_type in (StructureType.ROAD, StructureType.ROAD_START):
                bonus += size * 1.0
            else:
                bonus += size * 1.5
            break
    return bonus

def evaluate_move(game, move, bot_state: BotState):
    is_winner = bot_state.is_winner(game.state)
    score = 0
    unclaimed_structure_bonus = add_bonus_for_unclaimed_structures(bot_state, move)
    print(f"Unclaimed structure bonus: {unclaimed_structure_bonus}")
    score += unclaimed_structure_bonus
    tile = move["tile"]
    # Select best location to place monastary based on how speed we can complete the structure
    if TileModifier.MONASTARY in tile.modifiers:
        neighbour_weighting = count_existing_neighbours_monastery(game, move) * 1.1
        score += (4.5 + neighbour_weighting)
    else:
        # Also consider if we can place tile on monastary we already own
        for monastary_point in bot_state.monastary_points:
            empty = 0
            if abs(move["tx"] - monastary_point[0]) <= 1 and abs(move["ty"] - monastary_point[1]) <= 1:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if game.state.map._grid[monastary_point[1] + dy][monastary_point[0] + dx] is None:
                            empty += 1
                score += 9 - empty
                break

    # Now deduct points for cases where we are helping opponent add to their monastary penalty will be based on how close meeple is from being freed
    score -= opponent_monastary_extension_penalty(game,bot_state,move)
    
    x, y = move["tx"], move["ty"]
    new_grid = [row[:] for row in game.state.map._grid]
    tile.placed_pos = (x, y)
    new_grid[y][x] = tile

    for edge in tile.get_external_tiles(new_grid).keys():
        structure_type, connected_parts, is_complete, unique_tiles = analyze_structure_from_edge(tile, edge, new_grid)
        structure_size = len(unique_tiles)
        multiplier = 1

        owners = game.state._get_claims(tile, edge)
        owner_penalty = 1

        if structure_type == StructureType.CITY:
            claim_bonus = city_extension_bonus(game, connected_parts)
            score += claim_bonus

            if is_complete:
                shield_bonus = get_shield_cities_count(unique_tiles) * 2
                base_score = structure_size * 2 + shield_bonus
                multiplier *= 1.5 if bot_state.obtain_max_points(game.state)*7/30 >= bot_state.meeples_placed else 1.0
                score += base_score * multiplier
                owner_penalty = score/2
            else:
                structure_type, connected_parts, is_complete, unique_tiles = analyze_structure_from_edge(tile, edge, game.state.map._grid)
                tile_points_needed = forecast_number_of_tiles_needed_to_complete(structure_type, unique_tiles, game.state.map._grid)
                tiles_remaining = given_structure_type_return_tiles_remaining(structure_type, game.state.map.available_tiles)
                probability = return_probability_given_comptaible_tiles(len(tiles_remaining), game.state.map.available_tiles)
                shield_bonus = get_shield_cities_count(unique_tiles) * 2
                score += (structure_size + shield_bonus) * probability
                owner_penalty = structure_size

            for owner in owners:
                if owner!=game.state.me.player_id:
                    score-=owner_penalty
                else:
                    score+=1

        elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
            if is_complete:
                multiplier *= 1.5 if bot_state.obtain_max_points(game.state)*7/30 >= bot_state.meeples_placed else 1.0
                score += structure_size * multiplier
                owner_penalty = score/2
            else:
                structure_type, connected_parts, is_complete, unique_tiles = analyze_structure_from_edge(tile, edge, game.state.map._grid)
                tile_points_needed = forecast_number_of_tiles_needed_to_complete(structure_type, unique_tiles, game.state.map._grid)
                tiles_remaining = given_structure_type_return_tiles_remaining(structure_type, game.state.map.available_tiles)
                probability = return_probability_given_comptaible_tiles(len(tiles_remaining), game.state.map.available_tiles)
                score += structure_size * probability
                owner_penalty = structure_size
            for owner in owners:
                if owner!=game.state.me.player_id:
                    score-=owner_penalty
                else:
                    score+=1

    print(score, flush=True)
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

def can_place_tile_at(game: Game, tile: Tile, x: int, y: int, rotation: int = 0) -> bool:
    grid = game.state.map._grid

    if not (0 <= y < len(grid) and 0 <= x < len(grid[0])):
        return False

    if grid[y][x] is not None:
        return False  # Already occupied

    tile_copy = deepcopy(tile)
    tile_copy.rotate_clockwise(rotation)

    directions = {
        (0, -1): "top_edge",
        (1, 0): "right_edge",
        (0, 1): "bottom_edge",
        (-1, 0): "left_edge",
    }

    edge_opposite = {
        "top_edge": "bottom_edge",
        "bottom_edge": "top_edge",
        "left_edge": "right_edge",
        "right_edge": "left_edge",
    }

    has_any_neighbour = False

    for (dx, dy), edge in directions.items():
        nx, ny = x + dx, y + dy

        if not (0 <= ny < len(grid) and 0 <= nx < len(grid[0])):
            continue

        neighbour_tile = grid[ny][nx]
        if neighbour_tile is None:
            continue

        has_any_neighbour = True

        if not StructureType.is_compatible(
            tile_copy.internal_edges[edge],
            neighbour_tile.internal_edges[edge_opposite[edge]],
        ):
            return False  # Edge mismatch

    if not has_any_neighbour:
        return False  # Must be adjacent to another tile

    # River rule validation (once per rotation)
    if any(edge == StructureType.RIVER for edge in tile_copy.internal_edges.values()):
        river_check = game.state.map.river_validation(tile_copy, x, y)
        if river_check != "pass":
            return False

    return True


def handle_place_tile(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
    print(f'Our monastaries {bot_state.monastary_points}')
    bot_state.move += 1
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    latest_tile = game.state.map.placed_tiles[-1]
    latest_pos = latest_tile.placed_pos
    assert latest_pos

    print(f'Structures we are tracking: {bot_state.claimed_structures}')
    # Now update the structures we tracking each turn
    bot_state.updated_claimed_structures(game)
    print(f'New set of structures we are tracking: {bot_state.claimed_structures}')

    directions = {(1, 0): "left_edge", (0, 1): "top_edge", (-1, 0): "right_edge", (0, -1): "bottom_edge"}

    # Check if river stage
    is_river = any(is_river_tile(tile) for tile in game.state.my_tiles)

    # Determine open candidate positions
    candidates = set()
    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] is None:
                        candidates.add((nx, ny))

    legal_moves = []
    bot_state.unclaimed_open_spots = large_unclaimed_structures(game, size_threshold=3)
    for tile_index, tile in enumerate(game.state.my_tiles):
        if is_river:
            print("In river stage", flush=True)
            for (dx, dy) in directions:
                tx, ty = latest_pos[0] + dx, latest_pos[1] + dy
                for rotation in range(4):
                    if can_place_tile_at(game, tile, tx, ty, rotation):
                        tile.rotate_clockwise(rotation)
                        tile.placed_pos = (tx, ty)
                        bot_state.last_tile = tile
                        return game.move_place_tile(query, tile._to_model(), tile_index)
            tile.rotate_clockwise(4)  # Reset tile

        else:
            print("In normal stage", flush=True)
            for (x, y) in candidates:
                for rotation in range(4):
                    if can_place_tile_at(game, tile, x, y, rotation):
                        legal_moves.append({
                            "tile": tile,
                            "rotation": rotation,
                            "tx": x,
                            "ty": y,
                            "index": tile_index
                        })

    if not is_river:
        if not legal_moves:
            return brute_force_tile(game, bot_state, query)

        move = get_best_evaluated_move(game, legal_moves, bot_state)
        tile = move["tile"]
        tile.rotate_clockwise(move["rotation"])
        tile.placed_pos = (move["tx"], move["ty"])
        bot_state.last_tile = tile

        if TileModifier.MONASTARY in tile.modifiers:
            bot_state.add_monastary(move["tx"], move["ty"])

        return game.move_place_tile(query, tile._to_model(), move["index"])


def brute_force_tile(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
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
                for dx, dy in directions:
                    x1, y1 = x + dx, y + dy
                    if not (0 <= x1 < width and 0 <= y1 < height):
                        continue
                    for tile_index, tile in enumerate(game.state.my_tiles):
                        for rotation in range(4):
                            if can_place_tile_at(game, tile, x1, y1, rotation):
                                tile.rotate_clockwise(rotation)
                                tile.placed_pos = (x1, y1)
                                bot_state.last_tile = tile
                                if TileModifier.MONASTARY in tile.modifiers:
                                    bot_state.add_monastary(x1, y1)
                                return game.move_place_tile(query, tile._to_model(), tile_index)
                        tile.rotate_clockwise(4)  # Reset after trying all rotations


def forecast_number_of_tiles_needed_to_complete(structure_type, tiles, grid):
    """
    Returns the locations of where we need tiles to complete structure
    """
    directions = {(0, -1): "top_edge",(1, 0): "right_edge",(0, 1): "bottom_edge",(-1, 0): "left_edge"}
    # Have a set tracking each tile we need based on location so we do not over-calculate
    tiles_needed = set()
    # Iterate through each tile after BFS and see how many we need
    for tile in tiles:
        # Go through each edge in the tile
        for direction, edge in directions.items():
            if tile.internal_edges.get(edge) == structure_type:
                x, y = tile.placed_pos
                tx, ty = x + direction[0], y + direction[1]
                # See if the location we are looking at is empty or connected and if it is in the set
                if grid[ty][tx] is None:
                    tiles_needed.add((tx,ty))
    print(f'Tiles needed: {tiles_needed}')
    return tiles_needed

def given_structure_type_return_tiles_remaining(structure_type,tiles_left):
    """
    Given structure type return all tiles left
    """
    # Now iterate each edge of tile to see if is the structure-type we need
    left = set()
    edges = ["top_edge","right_edge","bottom_edge","left_edge"]
    for tile in tiles_left:
        for edge in edges:
            if tile.internal_edges.get(edge) == structure_type:
                left.add(tile)
                break
    print(f'{structure_type} has {len(left)} left which are {left}')
    return left

def given_tiles_left_return_valid_tiles(tiles_left,adjacent_places,game):
    """
    Given empty locations where we can place tiles see the number of actual valid tiles we can use
    """
    tile_count = 0
    for tile in tiles_left:
        for x,y in adjacent_places:
            # For each location see if we can get a fit using tiles left based on given structure
            for rotation in range(4):
                if can_place_tile_at(game, tile, x, y, rotation):
                    tile_count += 1
                    break
    print(f'Predicted tiles needed {tile_count}')
    return tile_count

def return_probability_given_comptaible_tiles(compatible_tiles,tiles_left):
    if(len(tiles_left)==0):
        return 0
    probability_draw = compatible_tiles/(len(tiles_left))
    # Have this just in case there are more tiles than what is left
    probability_draw = min(1,probability_draw)
    return probability_draw

def compute_probability_of_monastary_completion(tile, game):
    empty = 0
    # Find number of empty tiles around monastary 
    for horizontal in range(-1,2):
        for vertical in range(-1,2):
            if game.state.map._grid[tile.placed_pos[1]+vertical][tile.placed_pos[0]+horizontal] is None:
                empty += 1
    empty -= 1
    filled = 9 - empty
    # Have a simple way of estimating monastary completion for now
    return ((filled)*0.75)/8
    
def evaluate_meeple_placement(structure_type: StructureType, bot_state: BotState, edge, tile, game):
    strategy_bonus = 2 if bot_state.strat_pref == 'A' else 0
    probability = 1
    potential_points = 0 

    if structure_type != StructureType.MONASTARY:
        # Now consider probability structure will be complete if we place meeple
        # # First run BFS on the edge 
        structure_type, connected_parts, is_complete, unique_tiles = analyze_structure_from_edge(tile,edge,game.state.map._grid)
        # Get tile locations which can be used to complete sturcture
        tile_points_for_structure = forecast_number_of_tiles_needed_to_complete(structure_type,unique_tiles, game.state.map._grid)
        # Given structure get all tiles
        tiles_given_structure = given_structure_type_return_tiles_remaining(structure_type, game.state.map.available_tiles)
        # Now given all available structure tiles find which ones are comptaible
        compatible_count = given_tiles_left_return_valid_tiles(tiles_given_structure,tile_points_for_structure,game)
        # Now given number of tiles compatible return probability
        probability = return_probability_given_comptaible_tiles(len(tiles_given_structure),game.state.map.available_tiles)
    else:
        probability = compute_probability_of_monastary_completion(tile,game)
    
    if structure_type == StructureType.CITY:
        # Compute if we have shield on CITY
        number_of_shields = get_shield_cities_count(unique_tiles)
        potential_points = len(unique_tiles) * 2 + number_of_shields * 2
    elif structure_type == StructureType.MONASTARY:
        potential_points = 9
    # Otherwise compute roads
    elif structure_type == StructureType.ROAD or structure_type == StructureType.ROAD_START:
        potential_points = len(unique_tiles)
    # This is for fields let us value it as 0 
    else:
        potential_points = 0

    # Late game make sure structures with lower probability are not as strong for consideration
    if probability <= 0.4 and bot_state.move >= 10:
        probability *= 0.75
    print(f'We are on move {bot_state.move}')

    # Determine meeple based on potential points * probability we can finish structure
    total_points = potential_points + potential_points * probability + strategy_bonus
    print(f"[INFO] Structure: {structure_type.name}, Points: {potential_points}, Prob: {probability:.2f}, Move: {bot_state.move}, Score: {total_points:.2f}")
    return (total_points)

def handle_place_meeple_advanced(game: Game, bot_state: BotState, query: QueryPlaceMeeple):
    if bot_state.last_tile is None or bot_state.meeples_placed == 7:
        return game.move_place_meeple_pass(query)
        
    structures = game.state.get_placeable_structures(bot_state.last_tile._to_model())
    tile_model = bot_state.last_tile
    bot_state.last_tile = None

    print(f'Structures we are tracking: {bot_state.claimed_structures}')
    # Now update the structures we tracking each turn
    bot_state.updated_claimed_structures(game)
    print(f'New set of structures we are tracking: {bot_state.claimed_structures}')
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
                score = evaluate_meeple_placement(structure, bot_state, edge, placed_tile, game)
                structure_scores.append((score, edge, structure))
                print(f"Added to consideration: Edge={edge}, Structure={structure}, Score={score}")
        
        print(f"Final structure_scores: {structure_scores}")
        
        if not structure_scores:
            print("No valid structures found, passing")
            return game.move_place_meeple_pass(query)
            
        best_meeple_placement = max(structure_scores, key=lambda x: x[0])
        best_score, best_edge, best_structure = best_meeple_placement[0], best_meeple_placement[1], best_meeple_placement[2]
       
        print(f"Best placement: Edge={best_edge}, Structure={best_structure}, Score={best_score}")
        
        if (best_score>4 or (1+bot_state.obtain_max_points(game.state)*7/30))>=bot_state.meeples_placed or TileModifier.MONASTARY in tile_model.modifiers or (is_completed and not is_claimed):
            bot_state.meeples_placed += 1
            # Add the structure to the ones we own
            bot_state.add_claimed_structure(best_structure,placed_tile,best_edge)
            return game.move_place_meeple(query, tile_model._to_model(), placed_on=best_edge)
            
    return game.move_place_meeple_pass(query)

if __name__ == "__main__":
    main()