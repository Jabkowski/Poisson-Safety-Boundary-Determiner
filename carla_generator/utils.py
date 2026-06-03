from enum import IntEnum


class CityObjectLabel(IntEnum):
    """Semantic tag IDs aligned with CARLA's CityObjectLabel enum."""

    NoneLabel = 0
    Buildings = 1
    Fences = 2
    Other = 3
    Pedestrians = 4
    Poles = 5
    RoadLines = 6
    Roads = 7
    Sidewalks = 8
    Vegetation = 9
    Vehicles = 10
    Walls = 11
    TrafficSigns = 12
    Sky = 13
    Ground = 14
    Bridge = 15
    RailTrack = 16
    GuardRail = 17
    TrafficLight = 18
    Static = 19
    Dynamic = 20
    Water = 21
    Terrain = 22


# RGB colors from CARLA's C++ CITYSCAPES_PALETTE_MAP for labels 0..23.
CITYSCAPES_PALETTE_MAP = [
    (0, 0, 0),
    (70, 70, 70),
    (100, 40, 40),
    (55, 90, 80),
    (220, 20, 60),
    (153, 153, 153),
    (157, 234, 50),
    (128, 64, 128),
    (244, 35, 232),
    (107, 142, 35),
    (0, 0, 142),
    (102, 102, 156),
    (220, 220, 0),
    (70, 130, 180),
    (81, 0, 81),
    (150, 100, 100),
    (230, 150, 140),
    (180, 165, 180),
    (250, 170, 30),
    (110, 190, 160),
    (170, 120, 50),
    (45, 60, 150),
    (145, 170, 100),
    (255, 0, 0),
]


CITYSCAPES_PALETTE_BY_LABEL = {
    label: CITYSCAPES_PALETTE_MAP[int(label)] for label in CityObjectLabel
}


def get_cityscapes_color(label):
    """Return RGB tuple for a semantic label id or CityObjectLabel."""
    return CITYSCAPES_PALETTE_MAP[int(label)]
