# combo_results_formatter.py

def summarize_combo_results(results):
    avg_reliability = sum([r["quality"]["reliability"] for r in results]) / len(results)
    overall_grade = "🟢 GOOD" if avg_reliability >= 95 else "🟡 FAIR" if avg_reliability >= 85 else "🔴 POOR"

    return {
        "average_reliability": round(avg_reliability, 2),
        "overall_grade": overall_grade,
        "strategy_count": len(results),
    }
