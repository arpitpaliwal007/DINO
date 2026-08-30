from ovva.analytics import TemporalAnalytics

def test_temporal_summary_tracks_dwell_and_peak():
    analytics = TemporalAnalytics(fps=10)
    analytics.observe(0, [1, 2], [.9, .8]); analytics.observe(1, [1], [.7])
    assert analytics.summary() == {"unique_tracks": 2, "track_dwell_seconds": {"1": .2, "2": .1}, "mean_track_confidence": .8, "peak_concurrent_tracks": 2}
