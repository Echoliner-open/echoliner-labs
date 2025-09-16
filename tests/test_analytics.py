from echoliner.analytics.metrics import uptime_ratio, summary


def test_uptime_ratio():
    events = [True, True, False, True]
    assert uptime_ratio(events) == 0.75


def test_summary():
    data = [1.0, 2.0, 3.0]
    stats = summary(data)
    assert stats["mean"] == 2.0
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
