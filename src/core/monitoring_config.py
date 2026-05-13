from __future__ import annotations

from typing import Dict, List, Optional


THEATERS: List[Dict] = [
    {
        "theater_id": "us_west",
        "label": "US West",
        "center_lat": 47.6,
        "center_lon": -122.3,
        "radius_km": 450.0,
        "adsb": {"lat": 47.6, "lon": -122.3, "dist_km": 300},
    },
    {
        "theater_id": "us_central",
        "label": "US Central",
        "center_lat": 39.0,
        "center_lon": -95.0,
        "radius_km": 650.0,
        "adsb": {"lat": 39.0, "lon": -95.0, "dist_km": 500},
    },
    {
        "theater_id": "us_east",
        "label": "US East",
        "center_lat": 40.7,
        "center_lon": -74.0,
        "radius_km": 450.0,
        "adsb": {"lat": 40.7, "lon": -74.0, "dist_km": 300},
    },
    {
        "theater_id": "europe",
        "label": "Europe",
        "center_lat": 50.0,
        "center_lon": 10.0,
        "radius_km": 900.0,
        "adsb": {"lat": 50.0, "lon": 10.0, "dist_km": 600},
    },
    {
        "theater_id": "middle_east",
        "label": "Middle East",
        "center_lat": 25.0,
        "center_lon": 45.0,
        "radius_km": 900.0,
        "adsb": {"lat": 25.0, "lon": 45.0, "dist_km": 600},
        "ais_bounding_box": [[12.0, 32.0], [32.0, 60.0]],
    },
    {
        "theater_id": "pacific",
        "label": "Pacific",
        "center_lat": 13.5,
        "center_lon": 144.8,
        "radius_km": 1100.0,
        "adsb": {"lat": 13.5, "lon": 144.8, "dist_km": 800},
        "ais_bounding_box": [[0.0, 105.0], [25.0, 150.0]],
    },
    {
        "theater_id": "black_sea",
        "label": "Black Sea",
        "center_lat": 44.0,
        "center_lon": 33.5,
        "radius_km": 500.0,
        "ais_bounding_box": [[40.0, 27.0], [48.0, 42.0]],
    },
    {
        "theater_id": "baltic",
        "label": "Baltic",
        "center_lat": 59.0,
        "center_lon": 20.0,
        "radius_km": 550.0,
        "ais_bounding_box": [[53.0, 10.0], [66.0, 30.0]],
    },
]


def adsb_watch_areas() -> List[Dict]:
    return [theater for theater in THEATERS if theater.get("adsb")]


def ais_bounding_boxes() -> List[List[List[float]]]:
    return [theater["ais_bounding_box"] for theater in THEATERS if theater.get("ais_bounding_box")]


def _distance_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    dlat = lat_a - lat_b
    dlon = lon_a - lon_b
    return ((dlat * dlat) + (dlon * dlon)) ** 0.5 * 111.0


def theater_for_point(
    lat: float,
    lon: float,
    previous_theater_id: Optional[str] = None,
    previous_lat: Optional[float] = None,
    previous_lon: Optional[float] = None,
) -> str:
    containing = []
    closest_id = "global"
    closest_distance = None
    previous_theater = None

    for theater in THEATERS:
        distance_km = _distance_km(lat, lon, theater["center_lat"], theater["center_lon"])
        if theater["theater_id"] == previous_theater_id:
            previous_theater = theater
        if distance_km <= theater["radius_km"]:
            containing.append((distance_km, theater["theater_id"]))
        if closest_distance is None or distance_km < closest_distance:
            closest_distance = distance_km
            closest_id = theater["theater_id"]

    if containing:
        containing.sort(key=lambda item: item[0])
        if (
            previous_theater is not None
            and previous_lat is not None
            and previous_lon is not None
            and _distance_km(lat, lon, previous_lat, previous_lon) <= 35.0
        ):
            previous_distance = _distance_km(
                lat,
                lon,
                previous_theater["center_lat"],
                previous_theater["center_lon"],
            )
            if previous_distance <= previous_theater["radius_km"]:
                return previous_theater_id
        return containing[0][1]

    return closest_id
