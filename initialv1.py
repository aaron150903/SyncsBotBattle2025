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
        self.strat_pref = 'B'
    
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

    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
        is_river = is_river_tile(tile_in_hand)

        for (dx, dy) in directions.keys():
            tx, ty = latest_pos[0] + dx, latest_pos[1] + dy

            for rotation in range(4):
                if simulate_validate_place_tile(game, tile_in_hand, rotation, tx, ty):
                    tile_in_hand.rotate_clockwise(rotation)
                    bot_state.last_tile = tile_in_hand
                    bot_state.last_tile.placed_pos = (tx, ty)
                    return game.move_place_tile(query, tile_in_hand._to_model(), tile_hand_index)

            tile_in_hand.rotate_clockwise(4)  # Reset tile to original orientation

    print("Could not place tile with strategy, using brute force...", flush=True)
    return brute_force_tile(game, bot_state, query)

def evaluate_meeple_placement(structure_type: StructureType, bot_state: BotState):
    structure_points = {StructureType.CITY: 3, StructureType.MONASTARY: 3, StructureType.ROAD: 2, StructureType.FIELD: 1}
    strategy_bonus = 2 if bot_state.strat_pref == 'A' else 0
    return (structure_points[structure_type] + strategy_bonus)

def handle_place_meeple_advanced(game: Game, bot_state: BotState, query: QueryPlaceMeeple):
    if bot_state.last_tile is None or bot_state.meeples_placed == 7:
        return game.move_place_meeple_pass(query)
    
    structures = game.state.get_placeable_structures(bot_state.last_tile._to_model())
    tile_model = bot_state.last_tile
    bot_state.last_tile = None

    if structures:
        structure_scores = []
        for edge, structure in structures.items():
            if game.state._get_claims(tile_model, edge) or game.state._check_completed_component(tile_model,edge):
                continue
            else:
                score = evaluate_meeple_placement(structure, bot_state)
                structure_scores.append((score, edge, structure))
        best_meeple_placement = max(structure_scores, key=lambda x: x[0])
        best_score, best_edge, best_structure = best_meeple_placement[0], best_meeple_placement[1], best_meeple_placement[2]
        if (bot_state.strat_pref == 'A' and best_score > 1):
            return game.move_place_meeple(query, tile_model, best_edge)
        elif best_score > 2:
            return game.move_place_meeple(query, tile_model, best_edge)
        else:
            return game.move_place_meeple_pass(query)
    return game.move_place_meeple_pass(query)

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