"""Analytics utilities for EchoLiner platform."""

from __future__ import annotations

from typing import Sequence, Dict


def uptime_ratio(events: Sequence[bool]) -> float:
    """Compute uptime ratio from a sequence of boolean events.

    Parameters
    ----------
    events: Sequence[bool]
        True for uptime, False for downtime.

    Returns
    -------
    float
        Ratio of uptime to total events. Returns 0.0 if no events.
    """
    if not events:
        return 0.0
    uptime = sum(1 for e in events if e)
    return uptime / len(events)


def summary(data: Sequence[float]) -> Dict[str, float]:
    """Return basic statistics (mean, min, max) for the data."""
    if not data:
        raise ValueError("Data sequence must not be empty")
    total = sum(data)
    return {
        "mean": total / len(data),
        "min": min(data),
        "max": max(data),
    }
