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
        self.stealable_structs = []
        self.in_stealing_mode = False
    
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

def city_extension_bonus(game, connected_parts, structure_size, bot_state, structure_type, has_our_claim):
    # Value City 2 and road 1
    if structure_type == StructureType.CITY:
        claim_bonus = structure_size * 2
    else:
        claim_bonus = structure_size 

    bonus = 0
    if has_our_claim:
        # Smooth scaling instead of hard threshold
        phase_multiplier = 1 + 0.05 * bot_state.move  # increases gradually
        phase_multiplier = min(1.75, phase_multiplier)
        bonus += claim_bonus * phase_multiplier

    return bonus


def city_helping_penalty(game, tile, edge, claims):
    if not claims:
        return 0
    if game.state.me.player_id in claims:
        return 0

    opponent_scores = [
        player.points
        for player in game.state.players.values()
        if player.player_id in claims
    ]
    if not opponent_scores:
        return 0

    strongest_opponent_points = max(opponent_scores)
    return strongest_opponent_points / 5


def opponent_monastary_extension_penalty(game, bot_state, move):
    x, y = move['tx'], move['ty']
    grid = game.state.map._grid
    location = None

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = x + dx, y + dy
            if grid[ny][nx] and TileModifier.MONASTARY in grid[ny][nx].modifiers:
                if (nx, ny) not in bot_state.monastary_points:
                    location = (nx, ny)
                    break

    if location:
        placed = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = location[0] + dx, location[1] + dy
                if grid[ny][nx] is not None:
                    placed += 1
        return 2 * (1 + (placed / 9))
    return 0

def corner_steal_bonus(move, game, bot_state, steal_value_threshold = 5):
    curr_x, curr_y = move['tx'], move['ty']
    curr_tile = move['tile']
    grid = game.state.map._grid
    directions = {(-1, -1): "right_edge", (-1, 1): "left_edge", (1, -1): "right_edge", (1, 1): "left_edge"}
    my_edge = {(-1, -1): "top_edge", (-1, 1): "top_edge", (1, -1): "bottom_edge", (1, 1): "bottom_edge"}
    steal_bonus = 0
    tile_steals = []
    for (dy, dx), neighbours_edge in directions.items():
        nx, ny = curr_x + dx, curr_y + dy
        if not (0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH):
                continue
        corner_tile = grid[ny][nx]
        if not corner_tile:
            continue
        my_tile_edge = my_edge[(dy, dx)]
        if curr_tile.internal_edges.get(my_tile_edge) != StructureType.CITY:
            continue
        if curr_tile.internal_edges.get(my_tile_edge) != corner_tile.internal_edges.get(neighbours_edge):
            continue
        _, _, _, unique_tiles = analyze_structure_from_edge(corner_tile, neighbours_edge, grid)
        curr_steal_value = len(unique_tiles) * 2
        steal_bonus += curr_steal_value
        if curr_steal_value >= steal_value_threshold:
            tile_steals.append((curr_tile, my_tile_edge, curr_steal_value))
            bot_state.in_stealing_mode = True
    bot_state.stealable_structs = sorted(tile_steals, key=lambda x: x[2])  
    return steal_bonus


def evaluate_game_position_and_get_multipliers(game, bot_state):
    my_id = game.state.me.player_id
    players = list(game.state.players.values())

    # Extract scores
    my_points = next(p.points for p in players if p.player_id == my_id)
    opponent_points = [p.points for p in players if p.player_id != my_id]

    # Calculate metrics
    avg_opponent_points = sum(opponent_points) / len(opponent_points)
    score_diff = my_points - avg_opponent_points

    # Scaling logic based on position
    if score_diff < -10:
        # Losing badly —> go aggressive
        return (1.5, 0.5)
    elif score_diff < -2.5:
        # Losing moderately —> boost bonus, lower penalty slightly
        return (1.1, 0.8)
    elif score_diff <= 2.5:
        # Balanced —> play normally
        return (1.0, 1.0)
    elif score_diff <= 10:
        # Slightly ahead —> small penalty increase
        return (1.0, 1.1)
    else:
        # Winning big —> be conservative, penalize risky moves
        return (0.85, 1.5)
    
def evaluate_move(game, move, bot_state):
    score = 0
    tile = deepcopy(move["tile"])
    tile.rotate_clockwise(move["rotation"])

    if TileModifier.MONASTARY in tile.modifiers:
        neighbour_weighting = count_existing_neighbours_monastery(game, move) * 1.1
        score += (4.5 + neighbour_weighting)
    else:
        for mp in bot_state.monastary_points:
            if abs(move["tx"] - mp[0]) <= 1 and abs(move["ty"] - mp[1]) <= 1:
                empty = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = mp[0] + dx, mp[1] + dy
                        if game.state.map._grid[ny][nx] is None:
                            empty += 1
                score += 9 - empty
                break

    score -= opponent_monastary_extension_penalty(game, bot_state, move)

    x, y = move["tx"], move["ty"]
    new_grid = [row[:] for row in game.state.map._grid]
    tile.placed_pos = (x, y)
    new_grid[y][x] = tile

    bonus_multiplier, penalty_multiplier = evaluate_game_position_and_get_multipliers(game,bot_state)

    for edge in tile.get_external_tiles(new_grid).keys():
        struct_type, parts, is_complete, unique_tiles = analyze_structure_from_edge(tile, edge, new_grid)
        claimants = game.state._get_claims(tile, edge)
        ownership = game.state.me.player_id in claimants
        size = len(unique_tiles)

        # Compute the penalty for a specific structure
        penalty = city_helping_penalty(game, tile, edge, claimants)

        if struct_type == StructureType.CITY:
            shield_bonus = get_shield_cities_count(unique_tiles) * (2.5 if bot_state.move >= 10 else 2)
            stealing_bonus = corner_steal_bonus(move, game, bot_state, 5)
            score += stealing_bonus
            base_score = size * 2 + shield_bonus
            if ownership:
                score += city_extension_bonus(game, parts, size, bot_state, struct_type, True)

                if is_complete:
                    multiplier = 1.5 if bot_state.move >= 10 else 1.0
                    score += base_score * multiplier * bonus_multiplier - penalty
                else:
                    tiles_needed = forecast_number_of_tiles_needed_to_complete(struct_type, unique_tiles, game.state.map._grid)
                    tiles_remaining = given_structure_type_return_tiles_remaining(struct_type, game.state.map.available_tiles)
                    probability = return_probability_given_comptaible_tiles(len(tiles_remaining), game.state.map.available_tiles)

                    # Adjust for completion closeness
                    if len(tiles_needed) <= 2:
                        probability *= 1.25
                    # If it is requiring more tiles futher reduce
                    elif len(tiles_needed) >= 3:
                        probability *= 0.75

                    # Penalize for impossible completions
                    if len(tiles_needed) > len(tiles_remaining):
                        score -= base_score 

                    else:
                        score += (base_score * probability * bonus_multiplier) - penalty * penalty_multiplier
            else:
                if len(claimants) > 0:
                    if is_complete:
                        score -= penalty * penalty_multiplier * (1.5 if bot_state.move >= 10 else 1.0)
                    else:
                        score -= penalty * penalty_multiplier
                    

        elif struct_type in [StructureType.ROAD, StructureType.ROAD_START]:
            base_score = size
            if is_complete:
                base_score *= (1.5 if bot_state.move >= 10 else 1.0) * bonus_multiplier
            else:
                prob = return_probability_given_comptaible_tiles(
                    len(given_structure_type_return_tiles_remaining(struct_type, game.state.map.available_tiles)),
                    game.state.map.available_tiles
                )
                base_score *= prob

            if ownership:
                score += city_extension_bonus(game, parts, size, bot_state, struct_type, True)
                score += base_score - penalty * penalty_multiplier
            else:
                score -= penalty * penalty_multiplier * (1.5 if is_complete and bot_state.move >= 10 else 1.25 if is_complete else 1)

    # Meeple scarcity scaling
    meeples_left = 7 - bot_state.meeples_placed
    scarcity_factor = 0.9 + 0.1 * meeples_left / 7
    score *= scarcity_factor

    # Add bonus for unclaimed structures
    score += add_bonus_for_unclaimed_structures(bot_state, move)

    print(f"Evaluated score: {score:.2f}", flush=True)
    return score


def large_unclaimed_structures(game, size_threshold=3):
    grid = game.state.map._grid
    seen = set()
    result = {}
    directions = {(0,-1):"top_edge", (1,0):"right_edge", (0,1):"bottom_edge", (-1,0):"left_edge"}

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            tile = grid[y][x]
            if not tile:
                continue
            for edge in directions.values():
                if (tile, edge) in seen:
                    continue
                stype, parts, is_complete, tiles = analyze_structure_from_edge(tile, edge, grid)
                if stype not in (StructureType.CITY, StructureType.ROAD, StructureType.ROAD_START):
                    continue
                for t2, e2 in parts:
                    seen.add((t2, e2))
                if is_complete or len(tiles) < size_threshold:
                    continue
                if any(game.state._get_claims(t2, e2) for t2, e2 in parts):
                    continue
                open_spots = forecast_number_of_tiles_needed_to_complete(stype, tiles, grid)
                result[(stype, frozenset(tiles))] = open_spots
    return result


def add_bonus_for_unclaimed_structures(bot_state, move):
    x, y = move['tx'], move['ty']
    bonus = 0
    for (stype, struct_tiles), open_spots in bot_state.unclaimed_open_spots.items():
        if (x, y) in open_spots:
            size = len(struct_tiles)
            base = size * (1.0 if stype in (StructureType.ROAD, StructureType.ROAD_START) else 2.0)
            if len(open_spots) <= 2:
                base *= 1.25  # bonus for easy-to-finish
            bonus += base
            break
    return bonus


def get_best_evaluated_move(game: Game, legal_moves: set, bot_state: BotState):
    best_move = list(legal_moves)[0]
    best_score = float('-inf')
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
    bot_state.meeples_placed = 7 - game.state.players_meeples[game.state.me.player_id]
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
    bot_state.meeples_placed = 7 - game.state.players_meeples[game.state.me.player_id]

    if bot_state.last_tile is None or bot_state.meeples_placed >= 7:
        return game.move_place_meeple_pass(query)
        
    structures = game.state.get_placeable_structures(bot_state.last_tile._to_model())
    tile_model = bot_state.last_tile
    bot_state.last_tile = None

    print(f'Structures we are tracking: {bot_state.claimed_structures}')
    # Now update the structures we tracking each turn
    bot_state.updated_claimed_structures(game)
    print(f'New set of structures we are tracking: {bot_state.claimed_structures}')
    print(f"Available structures: {structures}")  # Debug print

    if bot_state.in_stealing_mode and len(bot_state.stealable_structs) > 0 and bot_state.meeples_placed <= 6:
        for struct_tuple in bot_state.stealable_structs:
            curr_tile, steal_edge = struct_tuple[0], struct_tuple[1]
            if bot_state.last_tile == curr_tile:
                bot_state.in_stealing_mode = False
                bot_state.stealable_structs = []
                print(f"FOUND STEALING POSITION")
                return game.move_place_meeple(query, curr_tile._to_model(), placed_on=steal_edge)
        bot_state.in_stealing_mode = False
        bot_state.stealable_structs = []
    
    if structures:
        structure_scores = []
        
        x, y = tile_model.placed_pos
        placed_tile = game.state.map._grid[y][x]
        
        for edge, structure in structures.items():
            # Use the placed tile from the map for claims checking
            is_claimed = game.state._get_claims(placed_tile, edge)
            is_completed = game.state._check_completed_component(placed_tile, edge)
            if game.state.me.player_id in is_claimed:
                continue
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
        
        if (bot_state.strat_pref == 'A' and best_score > 1):
            # Add the structure to the ones we own
            bot_state.add_claimed_structure(best_structure,placed_tile,best_edge)
            return game.move_place_meeple(query, tile_model._to_model(), placed_on=best_edge)
        elif best_score > 2:
            # Add the structure to the ones we own
            bot_state.add_claimed_structure(best_structure,placed_tile,best_edge)
            return game.move_place_meeple(query, tile_model._to_model(), placed_on=best_edge)
        else:
            print(f"Score {best_score} not high enough, passing")
            return game.move_place_meeple_pass(query)
            
    return game.move_place_meeple_pass(query)

if __name__ == "__main__":
    main()