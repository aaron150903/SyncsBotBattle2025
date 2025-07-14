from lib.interact.tile import Tile
from lib.interface.queries.query_place_tile import QueryPlaceTile
from lib.interface.queries.query_place_meeple import QueryPlaceMeeple
from lib.interface.queries.typing import QueryType
from lib.interface.events.moves.move_place_meeple import (
    MovePlaceMeeple,
    MovePlaceMeeplePass,
)
from lib.interface.events.moves.move_place_tile import MovePlaceTile
from lib.interface.events.moves.typing import MoveType
from helper.client_state import ClientSate
from helper.state_mutator import StateMutator
from helper.interface import Connection
from lib.models.tile_model import TileModel


class Game:
    def __init__(self) -> None:
        self.state = ClientSate()
        self.mutator = StateMutator(self.state)
        self.connection = Connection()

    def get_next_query(self) -> QueryType:
        query = self.connection.get_next_query()

        new_events_mark = len(self.state.event_history)
        for i, record in query.update.items():
            self.mutator.commit(i, record)
        self.state.new_events = new_events_mark

        return query

    def send_move(self, move: MoveType) -> None:
        self.connection.send_move(move)

    def move_place_tile(
        self, query: QueryPlaceTile, tile: TileModel, tile_index: int
    ) -> MovePlaceTile:
        return MovePlaceTile(
            player_id=self.state.me.player_id, tile=tile, player_tile_index=tile_index
        )

    def move_place_meeple(
        self, query: QueryPlaceMeeple, tile: TileModel, placed_on: str
    ) -> MovePlaceMeeple:
        return MovePlaceMeeple(
            player_id=self.state.me.player_id, tile=tile, placed_on=placed_on
        )

    def move_place_meeple_pass(self, query: QueryPlaceMeeple) -> MovePlaceMeeplePass:
        return MovePlaceMeeplePass(
            player_id=self.state.me.player_id,
        )

    def can_place_tile_at(self, tile: Tile, x: int, y: int) -> bool:
        if self.state.map._grid[y][x]:
            return False  # Already occupied

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

        print(f"Checking if tile can be placed {x, y}")
        has_any_neighbour = False

        for _ in range(4):  # Try all 4 rotations
            has_any_neighbour = False  # reset for each rotation

            for (dx, dy), edge in directions.items():
                nx, ny = x + dx, y + dy

                print(
                    f"Checking if tile neighbour compatible - {nx, ny} with rotation {tile.rotation}"
                )

                if not (
                    0 <= ny < len(self.state.map._grid)
                    and 0 <= nx < len(self.state.map._grid[0])
                ):
                    continue

                neighbour_tile = self.state.map._grid[ny][nx]

                if neighbour_tile is None:
                    continue

                has_any_neighbour = True
                # print(tile.internal_edges[edge], edge, tile.rotation, tile.tile_type)
                # print(neighbour_tile.internal_edges[edge_opposite[edge]])
                if (
                    tile.internal_edges[edge]
                    != neighbour_tile.internal_edges[edge_opposite[edge]]
                ):
                    print("Edge Missmatch")
                    break  # mismatch, try next rotation

            else:
                if has_any_neighbour:
                    print("Returning True")
                    return True

            tile.rotate_clockwise(1)

        return False
