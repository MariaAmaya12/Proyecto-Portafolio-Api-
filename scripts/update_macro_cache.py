import json
from datetime import datetime, timezone
import math

from src.api.macro import macro_snapshot


def clean_value(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return xf


def main():
    macro = macro_snapshot()

    data = {
        "risk_free_rate_pct": clean_value(macro.get("risk_free_rate_pct")),
        "inflation_yoy": clean_value(macro.get("inflation_yoy")),
        "cop_per_usd": clean_value(macro.get("cop_per_usd")),
        "usdcop_market": clean_value(macro.get("usdcop_market")),
        "source": "github_actions_cache",
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    with open("data/macro_cache.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()