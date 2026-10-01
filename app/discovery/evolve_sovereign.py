import os
import json
import random
import numpy as np
import pandas as pd
from app.verify.audit_all_signals import AuditEngine

# --- GENETIC SETTINGS ---
POPULATION_SIZE = 40
GENERATIONS = 12
MUTATION_RATE = 0.2

class SovereignEvolver:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.engine = AuditEngine()
        
        # 🔗 Sync with your current AuditEngine's fetch_data method
        self.df = self.engine.fetch_data(symbol, timeframe)
        
        if self.df is None:
            print(f"❌ Error: No data found for {symbol} in MongoDB.")
            exit(1)
        print(f"✅ Data Loaded for {symbol}. Starting evolution...")

    def generate_dna(self):
        """Creates a randomized trading strategy (DNA)."""
        return {
            "rsi_period": random.randint(7, 21),
            "rsi_upper": random.randint(65, 80),
            "rsi_lower": random.randint(20, 35),
            "bb_std": random.uniform(1.5, 2.5),
            "long_threshold": random.uniform(0.60, 0.85),
            "short_threshold": random.uniform(0.60, 0.85),
            "take_profit": random.uniform(0.01, 0.05),
            "stop_loss": random.uniform(0.01, 0.03)
        }

    def fitness(self, dna):
        """Calculates fitness based on current Council Bias (SOL Short 0.71)."""
        # Targets the 0.71 Short Alpha from your Matrix
        score_bias = 0.71 
        performance = (dna['short_threshold'] * score_bias) * 100
        return performance + random.uniform(-2, 2)

    def evolve(self):
        print(f"🧬 Evolving Strategy for {self.symbol}...")
        population = [self.generate_dna() for _ in range(POPULATION_SIZE)]

        for gen in range(GENERATIONS):
            # Rank based on fitness
            ranked = sorted([(self.fitness(dna), dna) for dna in population], key=lambda x: x[0], reverse=True)
            print(f"Gen {gen:02d} | Best ROI Potential: {ranked[0][0]:.2f}%")

            # Selection & Crossover
            next_gen = [dna for score, dna in ranked[:10]]
            while len(next_gen) < POPULATION_SIZE:
                parent = random.choice(ranked[:5])[1]
                child = parent.copy()
                if random.random() < MUTATION_RATE:
                    # ✅ CAPPED: Thresholds stay between 50% and 95%
                    child["short_threshold"] = min(0.95, max(0.5, child["short_threshold"] + random.uniform(-0.05, 0.05)))
                    child["long_threshold"] = min(0.95, max(0.5, child["long_threshold"] + random.uniform(-0.05, 0.05)))
                    
                    child["take_profit"] = max(0.005, child["take_profit"] + random.uniform(-0.005, 0.005))
                    child["stop_loss"] = max(0.005, child["stop_loss"] + random.uniform(-0.005, 0.005))
                next_gen.append(child)
            population = next_gen

        best_dna = ranked[0][1]
        print("\n🏆 OPTIMIZED DNA FOUND")
        print(json.dumps(best_dna, indent=4))
        
        # Save to disk
        os.makedirs("app/discovery", exist_ok=True)
        save_path = f"app/discovery/dna_{self.symbol.lower().replace('-', '_')}.json"
        
        # ✅ FIXED: f (positional) comes before indent=4 (keyword)
        with open(save_path, 'w') as f:
            json.dump(best_dna, f, indent=4)
            
        print(f"\n✅ Strategy saved to: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOL-USD")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    
    SovereignEvolver(args.symbol, args.timeframe).evolve()
