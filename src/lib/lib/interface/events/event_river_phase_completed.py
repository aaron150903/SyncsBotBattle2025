from lib.interface.events.base_event import BaseEvent

from typing import Literal

from lib.models.tile_model import TileModel


class EventRiverPhaseCompleted(BaseEvent):
    event_type: Literal["event_river_phase_completed"] = "event_river_phase_completed"
    end_tile: TileModel
