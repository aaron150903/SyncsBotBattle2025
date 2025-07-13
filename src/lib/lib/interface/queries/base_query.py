from lib.interface.events.typing import EventType

from typing import Mapping
from pydantic import BaseModel


class BaseQuery(BaseModel):
    query_type: str
    update: Mapping[int, EventType]
