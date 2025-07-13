from lib.interface.queries.base_query import BaseQuery

from typing import Literal


class QueryPlaceTile(BaseQuery):
    query_type: Literal["query_place_tile"] = "query_place_tile"
