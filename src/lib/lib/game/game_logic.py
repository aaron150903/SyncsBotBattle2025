from lib.config.map_config import MONASTARY_IDENTIFIER
from lib.interact.map import Map
from lib.interact.meeple import Meeple
from lib.interact.structure import StructureType
from lib.interact.tile import Tile, TileModifier

from collections import defaultdict, deque
from typing import Callable, Iterator, Protocol


class SharedGameState(Protocol):
    map: Map


class GameLogic(SharedGameState):
    def _get_claims_objs(self, tile: "Tile", edge: str) -> dict[int, list[Meeple]]:
        players = defaultdict(list)

        if edge == MONASTARY_IDENTIFIER:
            m = tile.internal_claims[edge]
            if not m:
                return {}

            return {m.player_id: [m]}

        for connected_tile, e in self._traverse_connected_component(tile, edge):
            meeple = connected_tile.internal_claims[e]
            if meeple is not None:
                players[meeple.player_id].append(meeple)

        return players

    def _get_claims(self, tile: "Tile", edge: str) -> list[int]:
        players: set[int] = set()

        if edge == MONASTARY_IDENTIFIER:
            m = tile.internal_claims[edge]
            if not m:
                return []

            return [m.player_id]

        for connected_tile, e in self._traverse_connected_component(tile, edge):
            meeple = connected_tile.internal_claims[e]
            if meeple is not None:
                players.add(meeple.player_id)

        return list(players)

    def _get_reward(self, tile: "Tile", edge: str, partial: bool = False) -> int:
        visited_tiles = set()
        structure_type = tile.internal_edges[edge]

        total_points = 0

        for connected_tile, _ in self._traverse_connected_component(tile, edge):
            if connected_tile in visited_tiles:
                continue

            visited_tiles.add(connected_tile)

            if not partial:
                total_points += StructureType.get_points(structure_type)

            else:
                total_points += StructureType.get_partial_points(structure_type)

            total_points = TileModifier.apply_point_modifiers(
                tile.modifiers, total_points
            )

        return total_points

    def _check_completed_component(self, start_tile: Tile, edge: str) -> bool:
        component = list(self._traverse_connected_component(start_tile, edge))

        for tile, edge in component:
            assert tile.placed_pos is not None
            if tile.get_external_tile(edge, tile.placed_pos, self.map._grid) is None:
                return False

        return True

    def check_any_complete(self, start_tile: "Tile") -> list[str]:
        edges_complete: list[str] = []
        for edge, tile in start_tile.get_external_tiles(self.map._grid).items():
            if tile and self._check_completed_component(start_tile, edge):
                edges_complete.append(edge)

        return edges_complete

    def _traverse_connected_component(
        self,
        start_tile: "Tile",
        edge: str,
        yield_cond: Callable[[Tile, str], bool] = lambda _1, _2: True,
        modify: Callable[[Tile, str], None] = lambda _1, _2: None,
    ) -> Iterator[tuple["Tile", str]]:
        visited = set()

        # Not a traversable edge - ie monastary etc
        if edge not in start_tile.internal_edges.keys():
            return

        structure_type = start_tile.internal_edges[edge]
        structure_bridge = TileModifier.get_bridge_modifier(structure_type)

        queue = deque([(start_tile, edge)])

        while queue:
            tile, edge = queue.popleft()

            if (tile, edge) in visited:
                continue

            # Visiting portion of traversal
            visited.add((tile, edge))
            modify(tile, edge)

            if yield_cond(tile, edge):
                yield tile, edge

            connected_internal_edges = [edge]

            for adjacent_edge in Tile.adjacent_edges(edge):
                if tile.internal_edges[adjacent_edge] == structure_type:
                    if not (
                        TileModifier.BROKEN_CITY in tile.modifiers
                        and structure_type == StructureType.CITY
                    ):
                        connected_internal_edges.append(adjacent_edge)

                        for adjacent_edge2 in Tile.adjacent_edges(adjacent_edge):
                            if (
                                tile.internal_edges[adjacent_edge]
                                == tile.internal_edges[adjacent_edge2]
                                and adjacent_edge2 not in connected_internal_edges
                            ):
                                connected_internal_edges.append(adjacent_edge2)

            if (
                len(connected_internal_edges) == 1
                and structure_bridge
                and structure_bridge in tile.modifiers
            ):
                if StructureType.is_compatible(
                    structure_type, tile.internal_edges[Tile.get_opposite(edge)]
                ):
                    connected_internal_edges.append(Tile.get_opposite(edge))

            if structure_type == StructureType.ROAD_START:
                structure_type = StructureType.ROAD

            for cid in connected_internal_edges:
                assert tile.placed_pos is not None
                neighbouring_tile = Tile.get_external_tile(
                    cid, tile.placed_pos, self.map._grid
                )

                if neighbouring_tile:
                    neighbouring_tile_edge = tile.get_opposite(cid)
                    neighbouring_structure_type = neighbouring_tile.internal_edges[
                        neighbouring_tile_edge
                    ]

                    if (
                        structure_type == StructureType.ROAD
                        and neighbouring_structure_type == StructureType.ROAD_START
                    ):
                        continue

                    if (neighbouring_tile, neighbouring_tile_edge) not in visited:
                        queue.append((neighbouring_tile, neighbouring_tile_edge))
