import numpy as np

def check_calibration(trades):
    if not trades:
        return "No trade data available."
    
    # Extract AI confidence at time of entry
    confidences = [t.get('signal_prob', 0.5) for t in trades]
    # Check if trade was profitable
    outcomes = [1 if t.get('pnl', 0) > 0 else 0 for t in trades]
    
    avg_confidence = np.mean(confidences)
    actual_win_rate = np.mean(outcomes)
    
    calibration_error = avg_confidence - actual_win_rate
    
    print(f"📊 COUNCIL CALIBRATION REPORT")
    print(f" - Avg. AI Confidence: {avg_confidence:.2%}")
    print(f" - Actual Win Rate:    {actual_win_rate:.2%}")
    
    if calibration_error > 0.15:
        print("⚠️ ALERT: Council is OVERCONFIDENT. Re-train the Stacking Judge.")
    elif calibration_error < -0.15:
        print("⚠️ ALERT: Council is UNDERCONFIDENT. Potential edge is being missed.")
    else:
        print("✅ Council is perfectly calibrated.")

# In your execute_backtest return, you can call this:
# check_calibration(mgr.trades)
