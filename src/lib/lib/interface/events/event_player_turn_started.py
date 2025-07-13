from lib.interface.events.base_event import BaseEvent

from typing import Literal


class EventPlayerTurnStarted(BaseEvent):
    event_type: Literal["event_player_turn_started"] = "event_player_turn_started"
    player_id: int
