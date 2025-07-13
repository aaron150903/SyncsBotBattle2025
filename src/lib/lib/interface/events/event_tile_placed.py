from lib.interface.events.base_event import BaseEvent

from typing import Literal

from lib.models.tile_model import TileModel


class EventStartingTilePlaced(BaseEvent):
    event_type: Literal["event_starting_tile_placed"] = "event_starting_tile_placed"
    tile_placed: TileModel
