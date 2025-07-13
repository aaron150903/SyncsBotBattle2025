from pydantic import Field
from lib.interface.events.moves.move_place_tile import (
    MovePlaceTile,
    PublicMovePlaceTile,
)
from lib.interface.events.moves.move_place_meeple import (
    MovePlaceMeeple,
    MovePlaceMeeplePass,
)

from typing import Annotated, Union, TypeAlias

MoveType: TypeAlias = Annotated[
    Union[
        MovePlaceTile,
        PublicMovePlaceTile,
        MovePlaceMeeple,
        MovePlaceMeeplePass,
    ],
    Field(discriminator="event_type"),
]
