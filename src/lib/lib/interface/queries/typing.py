from lib.interface.queries.query_place_tile import QueryPlaceTile
from lib.interface.queries.query_place_meeple import QueryPlaceMeeple

from pydantic import Field, RootModel
from typing import Annotated, TypeAlias, Union


QueryType: TypeAlias = Annotated[
    Union[
        QueryPlaceTile,
        QueryPlaceMeeple,
    ],
    Field(discriminator="query_type"),
]


class QueryTypeAdapter(RootModel[QueryType]):
    pass
