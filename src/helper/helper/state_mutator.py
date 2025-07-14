from helper.client_state import ClientSate

from lib.interact.meeple import Meeple
from lib.interface.events.event_player_bannned import EventPlayerBanned
from lib.interface.events.event_player_turn_started import EventPlayerTurnStarted
from lib.interface.events.event_player_won import EventPlayerWon
from lib.interface.events.event_river_phase_completed import EventRiverPhaseCompleted
from lib.interface.events.event_game_ended import (
    EventGameEndedCancelled,
    EventGameEndedPointLimitReached,
    EventGameEndedStaleMate,
)
from lib.interface.events.event_game_started import (
    EventGameStarted,
    PublicEventGameStarted,
)
from lib.interface.events.event_player_drew_tiles import (
    EventPlayerDrewTiles,
    PublicEventPlayerDrewTiles,
)
from lib.interface.events.event_player_meeple_freed import EventPlayerMeepleFreed
from lib.interface.events.event_tile_placed import (
    EventStartingTilePlaced,
)
from lib.interface.events.moves.move_place_meeple import (
    MovePlaceMeeple,
    MovePlaceMeeplePass,
)
from lib.interface.events.moves.move_place_tile import (
    MovePlaceTile,
    PublicMovePlaceTile,
)
from lib.interface.events.typing import EventType


class StateMutator:
    def __init__(self, state: ClientSate) -> None:
        self.state = state

    def commit(self, i: int, event: EventType) -> None:
        if i != len(self.state.event_history):
            raise RuntimeError("Please send us a discord message with this error log.")
        self.state.event_history.append(event)

        match event:
            case EventGameStarted() as e:
                self._commit_event_game_started(e)

            case PublicEventGameStarted() as e:
                self._commit_public_event_game_started(e)

            case EventPlayerDrewTiles() as e:
                self._commit_player_drew_tiles(e)

            case EventPlayerMeepleFreed() as e:
                self._commit_event_player_meeple_freed(e)

            case EventStartingTilePlaced() as e:
                self._commit_event_starting_tile_placed(e)

            case MovePlaceTile() as e:
                self._commit_move_place_tile(e)

            case PublicMovePlaceTile() as e:
                self._commit_public_move_place_tile(e)

            case MovePlaceMeeple() as e:
                self._commit_move_place_meeple(e)

            case MovePlaceMeeplePass() as e:
                self._commit_move_place_meeple_pass(e)

            case PublicEventPlayerDrewTiles() as e:
                self._commit_opponent_drew_tiles(e)

            case EventGameEndedPointLimitReached() as e:
                self._commit_event_game_ended_point_limit(e)

            case EventGameEndedStaleMate() as e:
                self._commit_event_game_ended_stalemate(e)

            case EventGameEndedCancelled() as e:
                self._commit_event_game_ended_cancelled(e)

            case EventPlayerBanned() as e:
                self._commit_event_player_banned(e)

            case EventPlayerTurnStarted() as e:
                self._commit_event_player_turn_started(e)

            case EventPlayerWon() as e:
                self._commit_event_player_won(e)

            case EventRiverPhaseCompleted() as e:
                self._commit_event_river_phase_completed(e)

            case _:
                raise RuntimeError(f"Unrecognised event: {event}")

    def _commit_player_drew_tiles(self, e: EventPlayerDrewTiles) -> None:
        if e.player_id != self.state.me.player_id:
            raise RuntimeError("Please send us a discord message with this error log.")

        self.state.me.tiles.extend(e.tiles)
        for tile_model in e.tiles:
            tile = self.state.map.get_tile_by_type(tile_model.tile_type, pop=True)

            self.state.my_tiles.append(tile)

    def _commit_opponent_drew_tiles(self, e: PublicEventPlayerDrewTiles) -> None:
        if e.player_id == self.state.me.player_id:
            raise RuntimeError("Please send us a discord message with this error log.")

        self.state.players[e.player_id].num_tiles += e.num_tiles

    def _commit_event_game_started(self, e: EventGameStarted) -> None:
        raise RuntimeError("Please send us a discord message with this error log.")

    def _commit_public_event_game_started(self, e: PublicEventGameStarted) -> None:
        self.state.me = e.you
        self.state.turn_order = e.turn_order
        self.state.players = {p.player_id: p for p in e.players}
        self.state.players_meeples = {
            p.player_id: e.num_starting_meeples for p in e.players
        }

        self.state.me.num_meeples = e.num_starting_meeples
        self.state.map.start_river_phase()

    def _commit_event_player_meeple_freed(self, e: EventPlayerMeepleFreed) -> None:
        x, y = e.tile.pos
        tile = self.state.map._grid[y][x]

        assert tile is not None
        tile.internal_claims[e.placed_on] = None
        self.state.players_meeples[e.player_id] += 1

        if e.player_id == self.state.me.player_id:
            self.state.me.num_meeples += 1

    def _commit_event_starting_tile_placed(self, e: EventStartingTilePlaced) -> None:
        self.state.map.place_river_start(e.tile_placed.pos)

    def _commit_move_place_tile(self, e: MovePlaceTile) -> None:
        if e.player_id != self.state.me.player_id:
            raise RuntimeError("Please send us a discord message with this error log.")

        x, y = e.tile.pos
        tile = self.state.my_tiles.pop(e.player_tile_index)
        tile.placed_pos = x, y

        self.state.map._grid[y][x] = tile
        self.state.map.placed_tiles.append(tile)
        self.state.players[e.player_id].num_tiles -= 1

        assert tile.rotation == e.tile.rotation

    def _commit_public_move_place_tile(self, e: PublicMovePlaceTile) -> None:
        self.state.players[e.player_id].num_tiles -= 1

        x, y = e.tile.pos
        tile = self.state.map.get_tile_by_type(e.tile.tile_type, pop=True)

        tile.placed_pos = x, y
        self.state.map._grid[y][x] = tile
        self.state.map.placed_tiles.append(tile)
        while tile.rotation != e.tile.rotation:
            tile.rotate_clockwise(1)

    def _commit_move_place_meeple(self, e: MovePlaceMeeple) -> None:
        self.state.players_meeples[e.player_id] -= 1

        x, y = e.tile.pos
        tile = self.state.map._grid[y][x]

        assert tile is not None
        tile.internal_claims[e.placed_on] = Meeple(e.player_id)

        if e.player_id == self.state.me.player_id:
            self.state.me.num_meeples -= 1

    def _commit_move_place_meeple_pass(self, e: MovePlaceMeeplePass) -> None:
        pass

    def _commit_event_game_ended_point_limit(
        self, e: EventGameEndedPointLimitReached
    ) -> None:
        pass

    def _commit_event_game_ended_stalemate(self, e: EventGameEndedStaleMate) -> None:
        pass

    def _commit_event_game_ended_cancelled(self, e: EventGameEndedCancelled) -> None:
        pass

    def _commit_event_player_banned(self, e: EventPlayerBanned) -> None:
        pass

    def _commit_event_player_turn_started(self, e: EventPlayerTurnStarted) -> None:
        pass

    def _commit_event_player_won(self, e: EventPlayerWon) -> None:
        pass

    def _commit_event_river_phase_completed(self, e: EventRiverPhaseCompleted) -> None:
        self.state.map.place_river_end(e.end_tile.pos, e.end_tile.rotation)
        self.state.map.start_base_phase()
