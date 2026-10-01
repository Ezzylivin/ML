import subprocess
import json
import re
import sys

# --- CONFIGURATION ---
assets = ["BTC", "ETH", "SOL", "XRP", "PEPE"]
results = []

print("\n📡 SCANNING SOVEREIGN ALPHA (Bi-Directional)...")
print("-" * 65)

for asset in assets:
    # Use the bi-directional audit script
    cmd = ["python3", "app/verify/audit_all_signals.py", "--symbol", f"{asset}-USD", "--timeframe", "1h"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout
    
    # REGEX: Extract the COUNCIL AVG line
    # Format expected: COUNCIL AVG | 0.37 | 0.11 | 💤 NO TRADE
    match = re.search(r"COUNCIL AVG\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)", output)
    
    if match:
        short_score = float(match.group(1))
        long_score = float(match.group(2))
        
        # Determine primary bias and confidence
        if short_score > long_score:
            bias = "SHORT"
            confidence = short_score
            emoji = "🔴"
        else:
            bias = "LONG"
            confidence = long_score
            emoji = "🟢"
            
        results.append({
            "asset": asset,
            "short": short_score,
            "long": long_score,
            "bias": bias,
            "confidence": confidence,
            "emoji": emoji
        })
    else:
        # Check for data errors (like PEPE missing data)
        if "Error" in output:
            print(f"⚠️  {asset:<5} : Data Missing/Insufficient")

# Rank results by absolute confidence (highest probability first)
results.sort(key=lambda x: x['confidence'], reverse=True)

print("\n🏆 RANKED COUNCIL VERDICTS (1h)")
print("=" * 65)
print(f"{'ASSET':<8} | {'BIAS':<8} | {'CONFIDENCE':<12} | {'S-AVG':<7} | {'L-AVG'}")
print("-" * 65)

for r in results:
    # Highlight high-conviction signals (> 0.60)
    alert = "⚡" if r['confidence'] > 0.60 else "  "
    print(f"{r['emoji']} {r['asset']:<5} {alert} | "
          f"{r['bias']:<8} | "
          f"{r['confidence']:<12.2f} | "
          f"{r['short']:<7.2f} | "
          f"{r['long']:.2f}")

print("=" * 65)

# Final Logic Recommendation
if results and results[0]['confidence'] > 0.60:
    top = results[0]
    print(f"\n🎯 ALPHA SIGNAL DETECTED: {top['asset']} {top['bias']} at {top['confidence']:.2f}")
    print(f"👉 Recommended Action: Run Genetic Evolver for {top['asset']}-USD")
else:
    print("\n💤 NO HIGH-CONVICTION SIGNALS: Market is in noise/neutrality.")
