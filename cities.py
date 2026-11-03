"""Kalshi daily-high cities used for the cross-city efficiency and liquidity study.

Stations come from the contracts' own rules: before the switch to The Weather Company (Aug 14,
2026) they name the NWS Climatological Report, and after it they name the CLI product (CLIMDW,
CLIDFW, ...). collect/city_markets.py checks each mapping against that station's ASOS observations.
The window is the stretch every city traded under NWS settlement.
"""
import datetime as dt
from collections import namedtuple

City = namedtuple('City', 'code series icao asos timezone name')

WINDOW_START = dt.date(2026, 4, 1)  # Newest series (KXHIGHT*) start here.
WINDOW_END = dt.date(2026, 8, 13)   # Last day before every series moved to The Weather Company.

CITIES = [
    City('NYC', 'KXHIGHNY', 'KNYC', 'NYC', 'America/New_York', 'New York (Central Park)'),
    City('LAX', 'KXHIGHLAX', 'KLAX', 'LAX', 'America/Los_Angeles', 'Los Angeles Airport'),
    City('CHI', 'KXHIGHCHI', 'KMDW', 'MDW', 'America/Chicago', 'Chicago Midway'),
    City('MIA', 'KXHIGHMIA', 'KMIA', 'MIA', 'America/New_York', 'Miami'),
    City('AUS', 'KXHIGHAUS', 'KAUS', 'AUS', 'America/Chicago', 'Austin Bergstrom'),
    City('DEN', 'KXHIGHDEN', 'KDEN', 'DEN', 'America/Denver', 'Denver'),
    City('PHL', 'KXHIGHPHIL', 'KPHL', 'PHL', 'America/New_York', 'Philadelphia'),
    City('SFO', 'KXHIGHTSFO', 'KSFO', 'SFO', 'America/Los_Angeles', 'San Francisco'),
    City('ATL', 'KXHIGHTATL', 'KATL', 'ATL', 'America/New_York', 'Atlanta'),
    City('SEA', 'KXHIGHTSEA', 'KSEA', 'SEA', 'America/Los_Angeles', 'Seattle'),
    City('BOS', 'KXHIGHTBOS', 'KBOS', 'BOS', 'America/New_York', 'Boston'),
    City('DFW', 'KXHIGHTDAL', 'KDFW', 'DFW', 'America/Chicago', 'Dallas-Fort Worth'),
    City('DCA', 'KXHIGHTDC', 'KDCA', 'DCA', 'America/New_York', 'Washington DC'),
    City('LAS', 'KXHIGHTLV', 'KLAS', 'LAS', 'America/Los_Angeles', 'Las Vegas'),
    City('MSP', 'KXHIGHTMIN', 'KMSP', 'MSP', 'America/Chicago', 'Minneapolis'),
]
BY_CODE = {c.code: c for c in CITIES}


def standard_offset(city):
    """UTC offset of local standard time; the NWS climate day runs midnight to midnight standard time."""
    from zoneinfo import ZoneInfo
    return ZoneInfo(city.timezone).utcoffset(dt.datetime(2026, 1, 15))


def day_start_utc(city, day):
    return dt.datetime.combine(day, dt.time(0), dt.timezone(standard_offset(city))).astimezone(dt.timezone.utc)
