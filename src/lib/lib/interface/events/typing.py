from pydantic import Field
from lib.interface.events.event_game_ended import (
    EventGameEndedCancelled,
    EventGameEndedPointLimitReached,
    EventGameEndedStaleMate,
)
from lib.interface.events.event_tile_placed import (
    EventStartingTilePlaced,
)
from lib.interface.events.event_game_started import (
    EventGameStarted,
    PublicEventGameStarted,
)
from lib.interface.events.event_player_bannned import EventPlayerBanned
from lib.interface.events.event_player_drew_tiles import (
    EventPlayerDrewTiles,
    PublicEventPlayerDrewTiles,
)
from lib.interface.events.event_player_meeple_freed import EventPlayerMeepleFreed
from lib.interface.events.event_player_turn_started import EventPlayerTurnStarted
from lib.interface.events.event_player_won import EventPlayerWon
from lib.interface.events.event_river_phase_completed import EventRiverPhaseCompleted
from lib.interface.events.moves.typing import MoveType

from typing import Annotated, TypeAlias, Union


EventType: TypeAlias = Annotated[
    Union[
        EventGameStarted,
        EventPlayerBanned,
        EventPlayerDrewTiles,
        EventPlayerTurnStarted,
        EventPlayerWon,
        EventRiverPhaseCompleted,
        EventPlayerMeepleFreed,
        EventPlayerWon,
        EventGameEndedCancelled,
        EventGameEndedPointLimitReached,
        EventGameEndedStaleMate,
        EventStartingTilePlaced,
        PublicEventGameStarted,
        PublicEventPlayerDrewTiles,
        MoveType,
    ],
    Field(discriminator="event_type"),
]
