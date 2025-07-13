from lib.config.scoring import NO_POINTS, ROAD_POINTS, CITY_POINTS

from enum import Enum, auto


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
