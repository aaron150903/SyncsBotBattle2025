from lib.interface.events.base_event import BaseEvent

from typing import Literal


class EventGameEndedPointLimitReached(BaseEvent):
    event_type: Literal["event_game_ended_point_limit_reached"] = (
        "event_game_ended_point_limit_reached"
    )
    player_id: int


class EventGameEndedStaleMate(BaseEvent):
    event_type: Literal["event_game_ended_stale_mate"] = "event_game_ended_stale_mate"
    reason: str


class EventGameEndedCancelled(BaseEvent):
    event_type: Literal["event_game_ended_cancelled"] = "event_game_ended_cancelled"
    reason: str
