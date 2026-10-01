# File: /root/Project/ML/data_quality_grader.py

import numpy as np

def grade_strategy_quality(metrics: dict, reproducibility_pass: bool, certification_pass: bool):
    """
    Assigns a data quality grade (GOOD / OKAY / BAD) based on backtest reliability.
    """
    calmar = metrics.get("calmar_ratio", 0)
    drawdown = metrics.get("max_drawdown", 100)
    win_rate = metrics.get("win_rate", 0.5)

    # Base reliability weighting
    reliability = (
        (np.clip(calmar / 3.0, 0, 1) * 0.4) +
        (1 - np.clip(drawdown / 50, 0, 1)) * 0.3 +
        (np.clip(win_rate, 0, 1) * 0.2) +
        (1.0 if reproducibility_pass else 0.0) * 0.05 +
        (1.0 if certification_pass else 0.0) * 0.05
    )

    reliability_pct = round(float(reliability * 100), 2)

    # Grade assignment
    if reliability_pct >= 90:
        grade = "🟢 GOOD"
    elif reliability_pct >= 75:
        grade = "🟡 OKAY"
    else:
        grade = "🔴 BAD"

    return {
        "grade": grade,
        "reliability": reliability_pct,
        "reproducibility_pass": reproducibility_pass,
        "certification_pass": certification_pass,
        "calmar_ratio": calmar,
        "max_drawdown": drawdown,
        "win_rate": win_rate,
    }
