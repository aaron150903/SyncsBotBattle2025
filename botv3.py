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
from typing import Callable, Iterator


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


def check_if_tile_completes_structure(game: Game, tile: Tile, x: int, y: int) -> bool:
    """
    Simulates placing a tile at (x, y) on a copied grid and checks if it would complete any connected structure.
    Does not check Monastery (Only roads and cities)
    """
    new_grid = [row[:] for row in game.state.map._grid]
    tile.placed_pos = (x, y)
    new_grid[y][x] = tile

    edges = tile.get_external_tiles(new_grid).keys()

    for edge in edges:
        component = list(transverse_connected_component(tile, edge, new_grid))
        for t, e in component:
            if t.placed_pos is None:
                return False
            if t.get_external_tile(e, t.placed_pos, new_grid) is None:
                return False
        if component:
            return True  # at least one structure is completed
    return False


def transverse_connected_component(
    start_tile: "Tile",
    edge: str,
    grid,
    yield_cond: Callable[[Tile, str], bool] = lambda _1, _2: True,
    modify: Callable[[Tile, str], None] = lambda _1, _2: None,
) -> Iterator[tuple["Tile", str]]:
    visited = set()

    if edge not in start_tile.internal_edges:
        return

    structure_type = start_tile.internal_edges[edge]
    structure_bridge = TileModifier.get_bridge_modifier(structure_type)
    queue = deque([(start_tile, edge)])

    while queue:
        tile, edge = queue.popleft()
        if (tile, edge) in visited:
            continue

        visited.add((tile, edge))
        modify(tile, edge)

        if yield_cond(tile, edge):
            yield tile, edge

        connected_internal_edges = [edge]

        for adjacent_edge in Tile.adjacent_edges(edge):
            if tile.internal_edges.get(adjacent_edge) == structure_type:
                if not (
                    TileModifier.BROKEN_CITY in tile.modifiers
                    and structure_type == StructureType.CITY
                ):
                    connected_internal_edges.append(adjacent_edge)

                    for adjacent_edge2 in Tile.adjacent_edges(adjacent_edge):
                        if (
                            tile.internal_edges.get(adjacent_edge2) == structure_type
                            and adjacent_edge2 not in connected_internal_edges
                        ):
                            connected_internal_edges.append(adjacent_edge2)

        if (
            len(connected_internal_edges) == 1
            and structure_bridge
            and structure_bridge in tile.modifiers
        ):
            if StructureType.is_compatible(
                structure_type,
                tile.internal_edges.get(Tile.get_opposite(edge))
            ):
                connected_internal_edges.append(Tile.get_opposite(edge))

        if structure_type == StructureType.ROAD_START:
            structure_type = StructureType.ROAD

        for cid in connected_internal_edges:
            if tile.placed_pos is None:
                continue

            neighbouring_tile = Tile.get_external_tile(cid, tile.placed_pos, grid)
            if neighbouring_tile:
                neighbouring_tile_edge = Tile.get_opposite(cid)
                neighbouring_structure_type = neighbouring_tile.internal_edges.get(neighbouring_tile_edge)

                if (
                    structure_type == StructureType.ROAD
                    and neighbouring_structure_type == StructureType.ROAD_START
                ):
                    continue

                if (neighbouring_tile, neighbouring_tile_edge) not in visited:
                    queue.append((neighbouring_tile, neighbouring_tile_edge))


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
                        tile.rotate_clockwise(rotation)
                        if check_if_tile_completes_structure(game, tile, x, y):
                            print('Structure Complete!',flush=True)
                            tile.placed_pos = (x, y)
                            bot_state.last_tile = tile
                            return game.move_place_tile(query, tile._to_model(), tile_index)
                        print('Structure Incomplete', flush=True)
                tile.rotate_clockwise(4)  # Reset rotation
    
    print("Could not place tile with strategy, using brute force...", flush=True)
    return brute_force_tile(game, bot_state, query)


def evaluate_meeple_placement(structure_type: StructureType, bot_state: BotState):
    if bot_state.strat_pref == 'A':
        structure_points = {StructureType.CITY: 3, StructureType.MONASTARY: 4, StructureType.ROAD: 2, StructureType.ROAD_START: 2, StructureType.GRASS: 0, StructureType.RIVER: 0}
    elif bot_state.strat_pref == 'D':
        structure_points = {StructureType.CITY: 4, StructureType.MONASTARY: 3, StructureType.ROAD: 2, StructureType.ROAD_START: 2, StructureType.GRASS: 0, StructureType.RIVER: 0}
    else:
        structure_points = {StructureType.CITY: 5, StructureType.MONASTARY: 4, StructureType.ROAD: 4, StructureType.ROAD_START: 3, StructureType.GRASS: 0, StructureType.RIVER: 0}
    strategy_bonus = 2 if bot_state.strat_pref == 'A' else 0
    total_points = (structure_points[structure_type] if structure_type in structure_points else 0) + strategy_bonus
    return (total_points)

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
        if not structure_scores:
            return game.move_place_meeple_pass(query) 
        best_meeple_placement = max(structure_scores, key=lambda x: x[0])
        best_score, best_edge, best_structure = best_meeple_placement[0], best_meeple_placement[1], best_meeple_placement[2]
       
        if (bot_state.strat_pref == 'A' and best_score > 1):
            # Update the meeples placed
            bot_state.meeples_placed += 1
            return game.move_place_meeple(query, tile_model._to_model(), placed_on=best_edge)
        elif best_score > 2:
            # Update the meeples placed
            bot_state.meeples_placed += 1
            return game.move_place_meeple(query, tile_model._to_model(), placed_on=best_edge)
        else:
            return game.move_place_meeple_pass(query)
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