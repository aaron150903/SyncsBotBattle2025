from engine.config.expansion_config import EXPANSION_PACKS
from engine.config.game_config import NUM_PLAYERS
from engine.game.tile_subscriber import TilePublisherBus
from engine.state.player_state import PlayerState
from engine.config.io_config import CORE_DIRECTORY

from lib.game.game_logic import GameLogic
from lib.interact.tile import Tile
from lib.interact.map import Map
from lib.interface.events.typing import EventType

import json


class GameState(GameLogic):
    def __init__(self) -> None:
        with open(f"{CORE_DIRECTORY}/input/catalog.json", "r") as f:
            self.catalog = json.load(f)

        self.round = -1
        self.players: dict[int, PlayerState] = {
            i: PlayerState(i, self.catalog[i]["team_id"]) for i in range(NUM_PLAYERS)
        }
        self.map = Map()

        self.game_over = False
        self.tiles_exhausted = True

        self.tile_placed: Tile | None = None
        self.tile_placed_claims: set[str] = set()
        self.tile_publisher = TilePublisherBus()

        self.event_history: list[EventType] = []
        self.turn_order: list[int] = []

        self.river_phase = True

    def _connect_players(self) -> None:
        for player in self.players.values():
            player.connect()

    def start_river_phase(self) -> None:
        self.map.start_river_phase()
        self.river_phase = True

    def start_base_phase(self) -> None:
        self.map.start_base_phase()

    def extend_base_phase(self) -> None:
        for ex in EXPANSION_PACKS:
            self.map.add_expansion_pack(None)

    def start_new_round(self) -> None:
        self.round += 1

    def get_player_points(self) -> list[tuple[int, int]]:
        return sorted(
            [(player.id, player.points) for player in self.players.values()],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_rankings(self) -> list[int]:
        return [
            player.id
            for player in sorted(
                self.players.values(), key=lambda p: p.points, reverse=True
            )
        ]

    def is_game_over(self) -> bool:
        return self.game_over

    def finalise_game(self) -> None:
        self.game_over = True
