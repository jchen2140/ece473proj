"""Persistence baseline: the latest known station high plus historical day-to-day changes.

Forecasts are issued at noon New York time on the day before the target date, when
the previous day's high is not yet final. The latest complete observation is therefore
the high from two days before the target (lag 2). The model is

    P(reported high = v) = share of 1991-2020 lag-2 changes (rounded high on day t minus
                           rounded high on day t-2), within +/-15 calendar days of the
                           target, that equal v - observed_high(target - 2 days).
"""
import datetime as dt
from collections import Counter

from collect_nyc import hits
from climatology import seasonal_distance

LAG_DAYS = 2


class Persistence:
    def __init__(self, history, start_year=1991, end_year=2020, window_days=15):
        self.end_year, self.window_days = end_year, window_days
        self.highs = {dt.date.fromisoformat(r['date']): int(r['rounded_tmax_f']) for r in history}
        self.changes = []
        for day, high in self.highs.items():
            before = day - dt.timedelta(days=LAG_DAYS)
            # Both days must be inside the training years so no later outcome leaks in.
            if start_year <= before.year and day.year <= end_year and before in self.highs:
                self.changes.append((day, high - self.highs[before]))

    def anchor(self, target):
        return self.highs.get(target - dt.timedelta(days=LAG_DAYS))

    def change_distribution(self, target):
        if target.year <= self.end_year:
            raise ValueError('Prediction year must follow the training period')
        counts = Counter(change for day, change in self.changes
                         if seasonal_distance(day, target) <= self.window_days)
        if not counts:
            raise ValueError('No historical changes in the seasonal window')
        return counts

    def probability(self, target, contract):
        """None if the lag-2 observation is missing."""
        base = self.anchor(target)
        if base is None:
            return None
        counts = self.change_distribution(target)
        total = sum(counts.values())
        return sum(n for change, n in counts.items() if hits(contract, base + change)) / total
