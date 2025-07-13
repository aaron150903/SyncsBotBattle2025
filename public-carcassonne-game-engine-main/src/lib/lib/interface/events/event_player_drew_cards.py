from lib.interface.events.base_event import BaseEvent

from typing import Literal

from lib.models.tile_model import TileModel


class EventPlayerDrewTiles(BaseEvent):
    event_type: Literal["event_player_drew_tiles"] = "event_player_drew_tiles"
    player_id: int
    num_tiles: int
    tiles: list[TileModel]

    def get_public(self) -> "PublicEventPlayerDrewTiles":
        return PublicEventPlayerDrewTiles(
            player_id=self.player_id, num_tiles=self.num_tiles
        )


class PublicEventPlayerDrewTiles(BaseEvent):
    event_type: Literal["public_event_player_drew_tiles"] = (
        "public_event_player_drew_tiles"
    )
    player_id: int
    num_tiles: int
