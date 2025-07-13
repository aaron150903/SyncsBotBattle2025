from pydantic import BaseModel

from lib.models.tile_model import TileModel


class PlayerModel(BaseModel):
    player_id: int
    team_id: int
    points: int
    tiles: list[TileModel]
    num_meeples: int

    def get_public(self) -> "PublicPlayerModel":
        return PublicPlayerModel(
            player_id=self.player_id, points=self.points, num_tiles=len(self.tiles)
        )


class PublicPlayerModel(BaseModel):
    player_id: int
    points: int
    num_tiles: int
