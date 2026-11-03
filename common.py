"""Paths, time zones and file helpers shared by every stage."""
import csv
import datetime as dt
import gzip
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
UTC = dt.timezone.utc
EST = dt.timezone(dt.timedelta(hours=-5))  # NWS climate day: fixed standard time
NY = ZoneInfo('America/New_York')


def write_csv(name, rows):
    path = DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_gz(path):
    import pandas as pd
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return pd.read_csv(f)
