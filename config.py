"""Single source of truth for the Milestone 1 study window and train/test split.

Every contract in the window names Central Park and the NWS Climatological Report.
Kalshi switched KXHIGHNY to The Weather Company starting with the Aug 14, 2026 event,
so the window stops at Aug 13, 2026. See DATA_NOTES.md for how these dates were chosen.
"""
import datetime as dt

WINDOW_START = dt.date(2025, 8, 14)
WINDOW_END = dt.date(2026, 8, 13)

# Bias/spread correction is fit on train days only; every reported table uses test days only.
TRAIN_START = dt.date(2025, 8, 14)
TRAIN_END = dt.date(2026, 4, 30)
TEST_START = dt.date(2026, 5, 1)
TEST_END = dt.date(2026, 8, 13)

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 473


def as_date(value):
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(value)


def split_of(day):
    """Return 'train', 'test', or None for a date (or ISO string) outside both periods."""
    day = as_date(day)
    if TRAIN_START <= day <= TRAIN_END:
        return 'train'
    if TEST_START <= day <= TEST_END:
        return 'test'
    return None


def window_days():
    return (WINDOW_END - WINDOW_START).days + 1
