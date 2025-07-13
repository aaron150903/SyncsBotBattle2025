from typing import TYPE_CHECKING

from engine.config.game_config import NUM_MEEPLES

from lib.interface.events.event_game_started import (
    EventGameStarted,
    PublicEventGameStarted,
)
from lib.interface.events.event_player_drew_tiles import EventPlayerDrewTiles
from lib.interface.events.moves.move_place_tile import MovePlaceTile
from lib.interface.events.typing import EventType

if TYPE_CHECKING:
    from engine.state.game_state import GameState


class CensorEvent:
    def __init__(self, state: "GameState") -> None:
        self.state = state

    def censor(self, event: EventType, player_id: int) -> EventType:
        match event:
            case MovePlaceTile() as e:
                if e.player_id == player_id:
                    return e

                return e.get_public()

            case EventPlayerDrewTiles() as e:
                if e.player_id == player_id:
                    return e

                return e.get_public()

            case EventGameStarted() as e:
                return PublicEventGameStarted(
                    turn_order=e.turn_order,
                    players=[player.get_public() for player in e.players],
                    num_starting_meeples=NUM_MEEPLES,
                    you=filter(
                        lambda x: x.player_id == player_id, e.players
                    ).__next__(),
                )

        return event
