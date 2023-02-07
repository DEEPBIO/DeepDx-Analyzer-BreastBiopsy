from typing import List, Literal, Union, TypedDict

Number = Union[int, float]

Coordinate = List[Number]  # ex) [0, 10000]


class RectOrEllipseData(TypedDict):
    min: Coordinate
    max: Coordinate
    angle: Number


class EllipseRegion(TypedDict):
    type: Literal["ellipse"]
    data: RectOrEllipseData


class RectangleRegion(TypedDict):
    type: Literal["rectangle"]
    data: RectOrEllipseData


class ContourRegion(TypedDict):
    type: Literal["contour"]
    data: List[List[Coordinate]]


Region = Union[RectangleRegion, EllipseRegion, ContourRegion]