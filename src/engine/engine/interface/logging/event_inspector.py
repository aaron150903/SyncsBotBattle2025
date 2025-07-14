from typing import Union
from engine.interface.io.game_result import (
    GameBanResult,
    GameCancelledResult,
    GameCrashedResult,
    GameSuccessResult,
)
from lib.interface.events.event_game_ended import (
    EventGameEndedCancelled,
)
from lib.interface.events.event_player_bannned import EventPlayerBanned
from lib.interface.events.event_player_won import EventPlayerWon
from lib.interface.events.typing import EventType
from lib.interface.events.event_game_started import EventGameStarted
from lib.interface.events.event_river_phase_completed import EventRiverPhaseCompleted
from lib.interface.events.event_tile_placed import (
    EventStartingTilePlaced,
)
from lib.interface.events.moves.move_place_meeple import (
    MovePlaceMeeple,
)
from lib.interface.events.moves.move_place_tile import MovePlaceTile

from pydantic import RootModel


class EventInspector:
    def __init__(
        self, history: list[EventType], points: dict[int, int], rankings: list[int]
    ) -> None:
        self.history = history
        self.points = points
        self.rankings = rankings

    def get_result(
        self,
    ) -> Union[
        GameBanResult, GameSuccessResult, GameCancelledResult, GameCrashedResult
    ]:
        match self.history[-1]:
            case EventGameEndedCancelled() as e:
                return GameCancelledResult(reason=e.reason)
            case EventPlayerBanned() as e:
                return GameBanResult(
                    ban_type=e.ban_type, player=e.player_id, reason=e.reason
                )
            case EventPlayerWon():
                return GameSuccessResult(rankings=self.rankings, points=self.points)
            case _:
                return GameCrashedResult(reason="Game engine crashed.")

    def get_recording_json(self) -> str:
        return RootModel(self.history).model_dump_json()

    def get_visualiser_json(self) -> str:
        visualiser_json: list[EventType] = []
        for i, event in enumerate(self.history):
            match event:
                case EventGameStarted() as e:
                    visualiser_json.append(e)

                case EventRiverPhaseCompleted() as e:
                    visualiser_json.append(e)

                case EventStartingTilePlaced() as e:
                    visualiser_json.append(e)

                case MovePlaceTile() as e:
                    visualiser_json.append(e)

                case MovePlaceMeeple() as e:
                    visualiser_json.append(e)

        return RootModel(visualiser_json).model_dump_json()
