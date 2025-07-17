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


class BotState:
    """A class for us to locally the state of the game and what we find relevant"""

    def __init__(self):
        self.last_tile: Tile | None = None
        self.meeples_placed: int = 0


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


def simulate_validate_place_tile(game, tile, rotation, x, y):
    """
    Simulate placement validation for tile before calling game.move_place_tile.
    Checks edge matching, river connection, U-turns, and adjacency.
    """
    from copy import deepcopy

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

def handle_place_tile(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
    """
    Handles placing a tile from hand by checking river phase and attempting valid moves.
    Uses simulated validation to avoid engine rejections.
    Stores all valid placements and picks the first one.
    """

    grid = game.state.map._grid
    latest_tile = game.state.map.placed_tiles[-1]
    latest_pos = latest_tile.placed_pos
    assert latest_pos

    directions = {
        (1, 0): "left_edge",
        (0, 1): "top_edge",
        (-1, 0): "right_edge",
        (0, -1): "bottom_edge",
    }

    valid_moves = []

    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):

        for (dx, dy) in directions.keys():
            tx, ty = latest_pos[0] + dx, latest_pos[1] + dy

            for rotation in range(4):
                if simulate_validate_place_tile(game, tile_in_hand, rotation, tx, ty):
                    valid_moves.append((tile_hand_index, tile_in_hand, rotation, (tx, ty)))

            tile_in_hand.rotate_clockwise(4)  # Reset tile to original orientation

    if valid_moves:
        print(f"Valid moves are {valid_moves}", flush=True)
        tile_hand_index, tile, rotation, pos = valid_moves[0]
        tile.rotate_clockwise(rotation)
        bot_state.last_tile = tile
        bot_state.last_tile.placed_pos = pos
        return game.move_place_tile(query, tile._to_model(), tile_hand_index)

    print("Could not place tile with strategy, using brute force...", flush=True)
    return brute_force_tile(game, bot_state, query)



def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """
    Attempts to place a meeple on the most recently placed tile.
    Falls back to passing if no valid placements are found.
    """
    # Pass placing a meeple if there is no tile placed or we have exceeded limit
    if bot_state.last_tile is None or bot_state.meeples_placed == 7:
        return game.move_place_meeple_pass(query)
    
    structures = game.state.get_placeable_structures(bot_state.last_tile._to_model())
    tile_model = bot_state.last_tile
    bot_state.last_tile = None

    if structures:
        for edge, _ in structures.items():
            if game.state._get_claims(tile_model, edge) or game.state._check_completed_component(tile_model,edge):
                continue
            else:
                # Update meeples palced
                bot_state.meeples_placed += 1
                return game.move_place_meeple(query, tile_model._to_model(), placed_on=edge)
            
    return game.move_place_meeple_pass(query)


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
                                return game.move_place_tile(query, tile._to_model(), tile_index)
                        tile.rotate_clockwise(4)  # Reset tile to original orientation


if __name__ == "__main__":
    main()


def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """
    Optimized meeple placement that maximizes expected scoring.
    Evaluates all possible placements and chooses the highest-scoring one.
    """
    # Pass if no tile placed or meeple limit reached
    if bot_state.last_tile is None or bot_state.meeples_placed == 7:
        return game.move_place_meeple_pass(query)
    
    tile_model = bot_state.last_tile
    structures = game.state.get_placeable_structures(tile_model._to_model())
    
    if not structures:
        bot_state.last_tile = None
        return game.move_place_meeple_pass(query)
    
    # Evaluate all valid placements
    placement_scores = []
    
    for edge, structure_info in structures.items():
        # Skip if already claimed or completed
        if game.state._get_claims(tile_model, edge) or game.state._check_completed_component(tile_model, edge):
            continue
            
        # Calculate expected score for this placement
        expected_score = calculate_placement_score(game, tile_model, edge, structure_info)
        placement_scores.append((expected_score, edge))
    
    # Clear the last tile reference
    bot_state.last_tile = None
    
    if not placement_scores:
        return game.move_place_meeple_pass(query)
    
    # Choose the highest-scoring placement
    placement_scores.sort(reverse=True)  # Sort by score descending
    best_score, best_edge = placement_scores[0]
    
    # Only place meeple if the expected score is worthwhile
    if best_score > get_minimum_score_threshold(bot_state.meeples_placed):
        bot_state.meeples_placed += 1
        return game.move_place_meeple(query, tile_model._to_model(), placed_on=best_edge)
    
    return game.move_place_meeple_pass(query)


def calculate_placement_score(game: Game, tile_model, edge: str, structure_info) -> float:
    """
    Calculate the expected score for placing a meeple on a specific structure.
    Returns a score estimate based on structure type and current state.
    """
    # Get the structure type from the tile's internal edges
    structure_type = tile_model.internal_edges[edge]
    
    match structure_type:
        case StructureType.ROAD | StructureType.ROAD_START:
            return calculate_road_score(game, tile_model, edge)
        case StructureType.CITY:
            return calculate_city_score(game, tile_model, edge)
        case StructureType.GRASS:
            return calculate_grass_score(game, tile_model, edge)
        case StructureType.MONASTARY:
            return calculate_monastery_score(game, tile_model, edge)
        case _:
            return 0.0


def calculate_road_score(game: Game, tile_model, edge: str) -> float:
    """
    Calculate expected score for road placement.
    Roads score ROAD_POINTS per tile segment when completed, 1 point per tile if incomplete at game end.
    """
    # Check if road is already completed (immediate scoring)
    if game.state._check_completed_component(tile_model, edge):
        # Get the length of the completed road
        road_length = get_structure_length(game, tile_model, edge)
        return road_length * ROAD_POINTS  # ROAD_POINTS per tile when completed
    
    # For incomplete roads, estimate based on current length and completion probability
    current_length = get_structure_length(game, tile_model, edge)
    completion_probability = estimate_completion_probability(game, tile_model, edge, StructureType.ROAD)
    
    # Expected score = (completion_prob * completed_score) + (incomplete_prob * incomplete_score)
    completed_score = current_length * ROAD_POINTS
    incomplete_score = current_length * 1.0  # 1 point per tile if incomplete at game end
    
    return (completion_probability * completed_score + 
            (1 - completion_probability) * incomplete_score)


def calculate_city_score(game: Game, tile_model, edge: str) -> float:
    """
    Calculate expected score for city placement.
    Cities score CITY_POINTS per tile segment when completed, 1 point per tile if incomplete at game end.
    """
    # Check if city is already completed (immediate scoring)
    if game.state._check_completed_component(tile_model, edge):
        # Get the length of the completed city
        city_length = get_structure_length(game, tile_model, edge)
        return city_length * CITY_POINTS  # CITY_POINTS per tile when completed
    
    # For incomplete cities, estimate based on current length and completion probability
    current_length = get_structure_length(game, tile_model, edge)
    completion_probability = estimate_completion_probability(game, tile_model, edge, StructureType.CITY)
    
    # Expected score = (completion_prob * completed_score) + (incomplete_prob * incomplete_score)
    completed_score = current_length * CITY_POINTS
    incomplete_score = current_length * 1.0  # 1 point per tile if incomplete at game end
    
    return (completion_probability * completed_score + 
            (1 - completion_probability) * incomplete_score)


def calculate_grass_score(game: Game, tile_model, edge: str) -> float:
    """
    Calculate expected score for grass (farm) placement.
    Farms score points based on completed cities they touch, only at game end.
    """
    # Farms are only scored at game end, so estimate based on adjacent cities
    adjacent_cities = count_adjacent_cities(game, tile_model, edge)
    
    # Estimate that cities have a moderate chance of being completed
    completion_probability = 0.6  # Adjustable based on game state analysis
    
    # Each completed city gives points to the farm (need to check actual farm scoring rules)
    # Assuming 3 points per completed city touching the farm
    expected_score = adjacent_cities * 3 * completion_probability
    
    # Farms are long-term investments, so apply a discount factor
    return expected_score * 0.7  # 70% value due to end-game only scoring


def calculate_monastery_score(game: Game, tile_model, edge: str) -> float:
    """
    Calculate expected score for monastery placement.
    Monasteries score 1 point + 1 per surrounding tile (max 9 points).
    """
    # Count current surrounding tiles
    surrounding_tiles = count_surrounding_tiles(game, tile_model)
    current_score = 1 + surrounding_tiles  # 1 base + 1 per surrounding tile
    
    # Estimate how many more tiles might be placed around the monastery
    empty_adjacent_spots = 8 - surrounding_tiles
    placement_probability = 0.4  # Conservative estimate for each spot
    
    expected_additional_tiles = empty_adjacent_spots * placement_probability
    expected_total_score = current_score + expected_additional_tiles
    
    return expected_total_score


def get_structure_length(game: Game, tile_model, edge: str) -> int:
    """
    Get the current length of a structure (road, river, etc.).
    This would need to traverse the connected structure across multiple tiles.
    """
    # This is a simplified version - actual implementation would need to
    # traverse the game map to find all connected tiles of the same structure
    # For now, return a conservative estimate
    return 2  # Placeholder - needs proper implementation


def count_adjacent_cities(game: Game, tile_model, edge: str) -> int:
    """
    Count how many cities are adjacent to a field.
    """
    # Placeholder implementation - would need to analyze the field's connections
    return 1  # Conservative estimate


def count_surrounding_tiles(game: Game, tile_model) -> int:
    """
    Count tiles surrounding a monastery.
    """
    if not tile_model.placed_pos:
        return 0
    
    x, y = tile_model.placed_pos
    grid = game.state.map._grid
    count = 0
    
    # Check all 8 surrounding positions
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # Skip the monastery tile itself
            
            nx, ny = x + dx, y + dy
            if (0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH and 
                grid[ny][nx] is not None):
                count += 1
    
    return count


def estimate_completion_probability(game: Game, tile_model, edge: str, structure_type) -> float:
    """
    Estimate the probability that a structure will be completed during the game.
    """
    # This is a simplified heuristic - could be made more sophisticated
    # by analyzing remaining tiles, open ends, etc.
    
    current_length = get_structure_length(game, tile_model, edge)
    
    # Shorter structures are more likely to be completed
    if current_length <= 3:
        return 0.8
    elif current_length <= 6:
        return 0.6
    else:
        return 0.3


def get_minimum_score_threshold(meeples_placed: int) -> float:
    """
    Return the minimum score threshold for placing a meeple.
    As meeples become scarce, require higher expected scores.
    """
    if meeples_placed <= 3:
        return 2.0  # Liberal placement early game
    elif meeples_placed <= 5:
        return 3.0  # More selective mid-game
    else:
        return 4.0  # Very selective late game