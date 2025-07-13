"""
Claude Bot - Strategic Carcassonne Player
A sophisticated bot that implements strategic gameplay for Carcassonne,
focusing on efficient scoring, blocking opponents, and optimal resource management.
"""

from helper.game import Game
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
from lib.interact.structure import StructureType
from typing import List, Tuple, Dict, Optional
import random


class BotState:
    """Maintains the bot's internal state and strategy decisions."""

    def __init__(self):
        self.last_tile: Optional[TileModel] = None
        self.strategy_preference = (
            "balanced"  # Can be "aggressive", "defensive", "balanced"
        )
        self.target_structures: List[
            str
        ] = []  # Structures we're focusing on completing
        self.blocked_positions: set = set()  # Positions we've intentionally blocked

    def update_strategy(self, game_state):
        """Dynamically adjust strategy based on game state."""
        my_points = game_state.points
        opponent_points = [
            p.points
            for p in game_state.players.values()
            if p.player_id != game_state.me.player_id
        ]

        if opponent_points and max(opponent_points) - my_points > 20:
            self.strategy_preference = "aggressive"
        elif my_points > 30:
            self.strategy_preference = "defensive"
        else:
            self.strategy_preference = "balanced"


def main():
    """Main game loop for the Claude bot."""
    game = Game()
    bot_state = BotState()

    while True:
        try:
            query = game.get_next_query()
            bot_state.update_strategy(game.state)

            def choose_move(query: QueryType) -> MoveType:
                match query:
                    case QueryPlaceTile() as q:
                        return handle_place_tile(game, bot_state, q)
                    case QueryPlaceMeeple() as q:
                        return handle_place_meeple(game, bot_state, q)

            move = choose_move(query)
            game.send_move(move)

        except Exception as e:
            print(f"Claude Bot Error: {e}")
            # Fallback to simple move if error occurs
            if isinstance(query, QueryPlaceTile):  # ignore type
                move = fallback_place_tile(game, bot_state, query)
            else:
                move = game.move_place_meeple_pass(query)
            game.send_move(move)


def handle_place_tile(
    game: Game, bot_state: BotState, query: QueryPlaceTile
) -> MovePlaceTile:
    """
    Strategic tile placement considering:
    1. Completing our own structures for maximum points
    2. Blocking opponent structures when beneficial
    3. Creating opportunities for future placements
    4. Maintaining board connectivity
    """

    # Get all possible placements and evaluate them
    possible_placements = get_all_valid_placements(game)

    if not possible_placements:
        # Fallback if no placements found
        return fallback_place_tile(game, bot_state, query)

    # Score each placement based on strategy
    scored_placements = []
    for tile_idx, tile, x, y, rotation in possible_placements:
        score = evaluate_tile_placement(game, bot_state, tile, x, y, rotation)
        scored_placements.append((score, tile_idx, tile, x, y, rotation))

    # Sort by score (highest first) and pick the best
    scored_placements.sort(reverse=True)
    best_score, best_tile_idx, best_tile, best_x, best_y, best_rotation = (
        scored_placements[0]
    )

    # Update bot state
    bot_state.last_tile = best_tile._to_model()
    bot_state.last_tile.pos = (best_x, best_y)
    bot_state.last_tile.rotation = best_rotation

    print(
        f"Claude placing tile {best_tile.tile_type} at ({best_x}, {best_y}) with rotation {best_rotation}, score: {best_score}"
    )

    return game.move_place_tile(query, best_tile, best_tile_idx)


def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """
    Strategic meeple placement considering:
    1. Potential points from completing structures
    2. Likelihood of structure completion
    3. Available meeples
    4. Opponent interference possibilities
    """

    if bot_state.last_tile is None:
        return game.move_place_meeple_pass(query)

    try:
        # Get available structures on the placed tile
        structures = game.state.get_tile_structures(bot_state.last_tile)

        if not structures:
            bot_state.last_tile = None
            return game.move_place_meeple_pass(query)

        # Filter out structures that are already claimed
        unclaimed_structures = get_unclaimed_structures(
            game, bot_state.last_tile, structures
        )

        if not unclaimed_structures:
            print("Claude: All structures on this tile are already claimed")
            bot_state.last_tile = None
            return game.move_place_meeple_pass(query)

        # Evaluate each unclaimed structure for meeple placement
        structure_scores = []
        for edge, structure_type in unclaimed_structures.items():
            score = evaluate_meeple_placement(game, bot_state, edge, structure_type)
            structure_scores.append((score, edge, structure_type))

        # Sort by score and consider placing on best structure
        structure_scores.sort(reverse=True)

        # Decision threshold - only place meeple if it's worth it
        best_score, best_edge, best_structure = structure_scores[0]

        # Check if we have available meeples
        my_meeples = game.state.get_meeples_placed_by(game.state.me.player_id)
        max_meeples = 7  # Standard Carcassonne meeple count

        if len(my_meeples) >= max_meeples:
            bot_state.last_tile = None
            return game.move_place_meeple_pass(query)

        # Place meeple if score is good enough or we're behind
        if best_score > 3 or (
            best_score > 1 and bot_state.strategy_preference == "aggressive"
        ):
            bot_state.target_structures.append(f"{bot_state.last_tile.pos}_{best_edge}")
            bot_state.last_tile = None
            print(
                f"Claude placing meeple on {best_edge} (structure: {best_structure}, score: {best_score})"
            )
            return game.move_place_meeple(
                query, bot_state.last_tile, placed_on=best_edge
            )

    except Exception as e:
        print(f"Meeple placement error: {e}")

    bot_state.last_tile = None
    return game.move_place_meeple_pass(query)


def get_all_valid_placements(game: Game) -> List[Tuple[int, object, int, int, int]]:
    """Get all valid tile placements for current hand."""
    placements = []
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    # Check all positions adjacent to existing tiles
    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:  # Found an existing tile
                # Check all adjacent positions
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    adj_x, adj_y = x + dx, y + dy

                    # Skip if position is out of bounds or occupied
                    if (
                        adj_x < 0
                        or adj_x >= width
                        or adj_y < 0
                        or adj_y >= height
                        or grid[adj_y][adj_x] is not None
                    ):
                        continue

                    # Try each tile in hand with each rotation
                    for tile_idx, tile in enumerate(game.state.my_cards):
                        original_rotation = tile.rotation
                        for rotation in range(4):
                            tile.rotation = rotation
                            if game.can_place_tile_at(tile, adj_x, adj_y):
                                placements.append(
                                    (tile_idx, tile, adj_x, adj_y, rotation)
                                )
                        tile.rotation = original_rotation  # Reset rotation

    return placements


def evaluate_tile_placement(
    game: Game, bot_state: BotState, tile, x: int, y: int, rotation: int
) -> float:
    """
    Evaluate the strategic value of a tile placement.
    Returns a score where higher is better.
    """
    score = 0.0

    # Base score for any valid placement
    score += 1.0

    # Bonus for completing structures
    completion_bonus = evaluate_structure_completion(game, tile, x, y)
    score += completion_bonus * 5.0

    # Bonus for extending our own structures
    extension_bonus = evaluate_structure_extension(game, tile, x, y)
    score += extension_bonus * 2.0

    # Penalty/bonus for blocking opponents
    blocking_value = evaluate_opponent_blocking(game, tile, x, y)
    if bot_state.strategy_preference == "aggressive":
        score += blocking_value * 3.0
    else:
        score += blocking_value * 1.0

    # Bonus for creating future opportunities
    future_potential = evaluate_future_potential(game, tile, x, y)
    score += future_potential * 1.5

    # Random tiebreaker
    score += random.random() * 0.1

    return score


def evaluate_structure_completion(game: Game, tile, x: int, y: int) -> float:
    """Evaluate if this placement completes any structures."""
    # This is a simplified version - in a full implementation,
    # you'd check if placing this tile completes roads, cities, or monasteries
    # For now, return a random bonus
    return random.uniform(0, 2)


def evaluate_structure_extension(game: Game, tile, x: int, y: int) -> float:
    """Evaluate if this placement extends our existing structures."""
    # Check if we have meeples on adjacent structures that this would extend
    my_meeples = game.state.get_meeples_placed_by(game.state.me.player_id)

    # Simplified: give bonus if we have nearby meeples
    bonus = 0.0
    for meeple in my_meeples:
        if meeple.placed and meeple.placed.placed_pos:
            meeple_x, meeple_y = meeple.placed.placed_pos
            distance = abs(meeple_x - x) + abs(meeple_y - y)
            if distance <= 2:  # Close to our meeples
                bonus += 1.0 / (distance + 1)

    return bonus


def evaluate_opponent_blocking(game: Game, tile, x: int, y: int) -> float:
    """Evaluate if this placement blocks opponent structures."""
    # Check for opponent meeples that this might block
    all_meeples = game.state.get_meeples_placed_by(None)
    opponent_meeples = [
        m for m in all_meeples if m.player_id != game.state.me.player_id
    ]

    blocking_value = 0.0
    for meeple in opponent_meeples:
        if meeple.placed and meeple.placed.placed_pos:
            meeple_x, meeple_y = meeple.placed.placed_pos
            distance = abs(meeple_x - x) + abs(meeple_y - y)
            if distance <= 2:  # Could potentially block
                blocking_value += 0.5

    return blocking_value


def evaluate_future_potential(game: Game, tile, x: int, y: int) -> float:
    """Evaluate the future potential of this placement."""
    # Give bonus for positions that have many open edges for future expansion
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    open_edges = 0
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        adj_x, adj_y = x + dx, y + dy
        if 0 <= adj_x < width and 0 <= adj_y < height and grid[adj_y][adj_x] is None:
            open_edges += 1

    return open_edges * 0.3


def evaluate_meeple_placement(
    game: Game, bot_state: BotState, edge: str, structure_type: StructureType
) -> float:
    """
    Evaluate the value of placing a meeple on a specific structure.
    Returns a score where higher is better.
    """
    score = 0.0

    # Base score varies by structure type
    if structure_type == StructureType.CITY:
        score += 3.0  # Cities are generally more valuable
    elif structure_type == StructureType.ROAD:
        score += 2.0  # Roads are medium value
    elif structure_type == StructureType.FIELD:
        score += 1.0  # Fields are lower immediate value
    else:
        score += 2.5  # Monasteries and others

    # Bonus for strategy preference
    if bot_state.strategy_preference == "aggressive":
        score += 1.0
    elif (
        bot_state.strategy_preference == "defensive"
        and structure_type == StructureType.CITY
    ):
        score += 2.0

    # Note: Structure claiming is now checked in get_unclaimed_structures()
    # before this function is called, so we don't need to check it here

    return score


def fallback_place_tile(
    game: Game, bot_state: BotState, query: QueryPlaceTile
) -> MovePlaceTile:
    """Fallback tile placement using simple strategy from simple.py."""
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    # Try to place any tile anywhere valid
    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                for tile_index, tile in enumerate(game.state.my_cards):
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        x1, y1 = x + dx, y + dy

                        if (
                            0 <= x1 < width
                            and 0 <= y1 < height
                            and grid[y1][x1] is None
                            and game.can_place_tile_at(tile, x1, y1)
                        ):
                            bot_state.last_tile = tile._to_model()
                            bot_state.last_tile.pos = (x1, y1)
                            return game.move_place_tile(query, tile, tile_index)

    # If we get here, something went wrong
    if game.state.my_cards:
        tile = game.state.my_cards[0]
        bot_state.last_tile = tile._to_model()
        bot_state.last_tile.pos = (0, 0)
        return game.move_place_tile(query, tile, 0)

    raise ValueError("No valid tile placement found and no cards available")


def get_unclaimed_structures(
    game: Game, tile_model: TileModel, structures: Dict[str, StructureType]
) -> Dict[str, StructureType]:
    """
    Filter structures to only return those that are not already claimed.
    This checks if a structure connects to an already claimed structure.
    """
    unclaimed = {}

    # Get the actual tile object from the grid
    x, y = tile_model.pos
    placed_tile = game.state.map._grid[y][x]

    if placed_tile is None:
        # Tile hasn't been placed yet, all structures are unclaimed
        return structures

    for edge, structure_type in structures.items():
        # Check if this edge/structure is already claimed by analyzing connected components
        if is_structure_unclaimed(game, placed_tile, edge):
            unclaimed[edge] = structure_type

    return unclaimed


def is_structure_unclaimed(game: Game, tile, edge: str) -> bool:
    """
    Check if a structure on a specific edge of a tile is unclaimed.
    Returns True if the structure can be claimed, False if it's already claimed.
    """
    try:
        # Check if there are existing meeples on connected structures
        # This is a simplified check - we look for meeples on the same tile edge
        if hasattr(tile, "internal_claims") and edge in tile.internal_claims:
            if tile.internal_claims[edge] is not None:
                return False  # This edge already has a meeple

        # Check adjacent tiles for connected structures with meeples
        # Get the position of the current tile
        if not hasattr(tile, "placed_pos") or tile.placed_pos is None:
            return True  # If tile position is unknown, assume unclaimed

        x, y = tile.placed_pos
        grid = game.state.map._grid

        # Check adjacent tiles based on the edge
        adjacent_positions = {
            "left_edge": (x - 1, y),
            "right_edge": (x + 1, y),
            "top_edge": (x, y - 1),
            "bottom_edge": (x, y + 1),
        }

        if edge in adjacent_positions:
            adj_x, adj_y = adjacent_positions[edge]

            # Check bounds
            if 0 <= adj_y < len(grid) and 0 <= adj_x < len(grid[0]):
                adjacent_tile = grid[adj_y][adj_x]

                if adjacent_tile is not None:
                    # Get the opposite edge on the adjacent tile
                    opposite_edges = {
                        "left_edge": "right_edge",
                        "right_edge": "left_edge",
                        "top_edge": "bottom_edge",
                        "bottom_edge": "top_edge",
                    }

                    opposite_edge = opposite_edges.get(edge)

                    # Check if structures match (roads connect to roads, cities to cities)
                    if hasattr(tile, "internal_edges") and hasattr(
                        adjacent_tile, "internal_edges"
                    ):
                        if (
                            edge in tile.internal_edges
                            and opposite_edge in adjacent_tile.internal_edges
                            and tile.internal_edges[edge]
                            == adjacent_tile.internal_edges[opposite_edge]
                        ):
                            # Structures match, check if adjacent structure is claimed
                            if (
                                hasattr(adjacent_tile, "internal_claims")
                                and opposite_edge in adjacent_tile.internal_claims
                                and adjacent_tile.internal_claims[opposite_edge]
                                is not None
                            ):
                                return False  # Connected structure is claimed

        return True  # Structure appears to be unclaimed

    except Exception as e:
        print(f"Error checking structure claims: {e}")
        return True  # Default to unclaimed if error occurs


if __name__ == "__main__":
    main()
