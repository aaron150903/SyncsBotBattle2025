from pydantic import BaseModel


class TileModel(BaseModel):
    tile_type: str
    pos: tuple[int, int]
    rotation: int
