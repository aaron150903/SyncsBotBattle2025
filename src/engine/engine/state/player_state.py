from engine.config.game_config import NUM_MEEPLES
from engine.interface.io.player_connection import PlayerConnection

from lib.interact.meeple import Meeple
from lib.interact.tile import Tile
from lib.models.player_model import PlayerModel


class PlayerState:
    def __init__(self, player_id: int, team_id: int) -> None:
        self.id = player_id
        self.team_id = team_id
        self.points = 0
        self.tiles: list[Tile] = []
        self.meeples: list["Meeple"] = [Meeple(player_id) for _ in range(NUM_MEEPLES)]
        self.connection: PlayerConnection

    def connect(self) -> None:
        self.connection = PlayerConnection(self.id)

    def _get_available_meeple(self) -> Meeple | None:
        available_meeples = [m for m in self.meeples if m.placed is None]

        if available_meeples:
            return available_meeples[0]

        return None

    def _to_player_model(self) -> PlayerModel:
        return PlayerModel(
            player_id=self.id,
            team_id=self.team_id,
            points=self.points,
            tiles=[tile._to_model() for tile in self.tiles],
            num_meeples=len([m for m in self.meeples if m.placed is None]),
        )
