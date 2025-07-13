from lib.interface.queries.base_query import BaseQuery

from typing import Literal


class QueryPlaceMeeple(BaseQuery):
    query_type: Literal["query_place_meeple"] = "query_place_meeple"
