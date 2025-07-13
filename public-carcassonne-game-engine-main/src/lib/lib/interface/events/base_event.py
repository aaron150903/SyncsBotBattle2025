from pydantic import BaseModel


class BaseEvent(BaseModel):
    event_type: str
