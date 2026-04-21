from src.signal import compute_signal


def test_bullish():
    assert compute_signal(close=110.0, sma=100.0, ema=105.0, rsi=55.0) == "Alcista"


def test_bearish():
    assert compute_signal(close=90.0, sma=100.0, ema=95.0, rsi=45.0) == "Bajista"


def test_neutral():
    assert compute_signal(close=110.0, sma=100.0, ema=105.0, rsi=54.9) == "Neutral"
