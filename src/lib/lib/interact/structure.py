from typing import TYPE_CHECKING, Optional
from lib.config.scoring import CITY_PARTIAL_POINTS, NO_POINTS, ROAD_POINTS, CITY_POINTS

from enum import Enum, auto

if TYPE_CHECKING:
    from lib.interact.tile import TileModifier


class StructureType(Enum):
    RIVER = auto()
    ROAD = auto()
    ROAD_START = auto()
    CITY = auto()
    GRASS = auto()

    MONASTARY = auto()

    @staticmethod
    def get_points(structure_type: "StructureType") -> int:
        return {
            StructureType.ROAD: ROAD_POINTS,
            StructureType.ROAD_START: ROAD_POINTS,
            StructureType.CITY: CITY_POINTS,
            # StructureType.GRASS: FARM_POINTS,
        }.get(structure_type, NO_POINTS)

    @staticmethod
    def get_partial_points(structure_type: "StructureType") -> int:
        return {
            StructureType.ROAD: ROAD_POINTS,
            StructureType.ROAD_START: ROAD_POINTS,
            StructureType.CITY: CITY_PARTIAL_POINTS,
            # StructureType.GRASS: FARM_POINTS,
        }.get(structure_type, NO_POINTS)

    @staticmethod
    def can_claim(structure_type: "StructureType") -> bool:
        return {
            StructureType.ROAD: True,
            StructureType.ROAD_START: True,
            StructureType.CITY: True,
            # StructureType.GRASS: True,
        }.get(structure_type, False)

    @staticmethod
    def is_compatible(
        s1: "StructureType",
        s2: "StructureType",
        m1: Optional[list["TileModifier"]] = None,
        m2: Optional[list["TileModifier"]] = None,
    ) -> bool:
        # Compatibility with ecternal edge
        return s2 in {
            StructureType.ROAD: [StructureType.ROAD, StructureType.ROAD_START],
            StructureType.ROAD_START: [StructureType.ROAD, StructureType.ROAD_START],
            StructureType.CITY: [StructureType.CITY],
            StructureType.RIVER: [StructureType.RIVER],
            StructureType.GRASS: [StructureType.GRASS],
        }.get(s1, [])
