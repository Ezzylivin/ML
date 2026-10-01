import json
import os
from datetime import datetime
from app.config2 import OPTIMIZER_DIR

class HallOfFame:
    """
    Saves and ranks the top-performing DNA Alphas.
    """
    @staticmethod
    def save_alpha(dna, metrics, symbol):
        path = os.path.join(OPTIMIZER_DIR, f"hall_of_fame_{symbol}.json")
        
        # 1. Load existing alphas
        alphas = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                alphas = json.load(f)

        # 2. Add new candidate
        new_entry = {
            "dna": dna,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "score": metrics.get('calmarRatio', 0)
        }
        alphas.append(new_entry)

        # 3. Rank and Prune (Keep top 20 Alphas)
        alphas.sort(key=lambda x: x['score'], reverse=True)
        alphas = alphas[:20]

        # 4. Atomic Save
        with open(path, 'w') as f:
            json.dump(alphas, f, indent=4)
        print(f"🏆 Alpha Certified & Saved to Hall of Fame: {dna}")
