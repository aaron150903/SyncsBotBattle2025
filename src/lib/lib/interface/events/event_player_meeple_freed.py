from lib.interface.events.base_event import BaseEvent

from typing import Literal

from lib.models.tile_model import TileModel


class EventPlayerMeepleFreed(BaseEvent):
    event_type: Literal["event_player_meeple_freed"] = "event_player_meeple_freed"
    player_id: int
    reward: int
    tile: TileModel
    placed_on: str
