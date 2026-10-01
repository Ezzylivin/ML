import random
import sys
import copy
import logging
import asyncio
import numpy as np
import pandas as pd  # ✅ ADDED MISSING IMPORT
import os            # ✅ ADDED FOR PATH FIX

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from datetime import datetime

# Internal Imports
from app.discovery.dna_logic import DNALogic
from app.discovery.hall_of_fame import HallOfFame

logger = logging.getLogger("GeneticEvolver")

class GeneticEvolver:
    def __init__(self, pop_size=50, mutation_rate=0.2):
        self.pop_size = max(pop_size, 10)
        self.mutation_rate = mutation_rate
        self.genes = DNALogic.get_base_genes()
        self.population = self._initialize_population()

    def _initialize_population(self):
        pop = []
        for _ in range(self.pop_size):
            pop.append({
                "dna": [
                    random.choice(self.genes["momentum"]), 
                    random.choice(self.genes["operators"]),
                    random.choice(self.genes["trend"]),
                    random.choice(self.genes["volatility"])
                ],
                "fitness": 0.0,
                "metrics": {}, 
                "generation": 0
            })
        return pop

    async def evolve_generation(self, backtest_func, df, symbol="BTC-USD"):
        tasks = [self._calculate_fitness(ind, backtest_func, df) for ind in self.population]
        await asyncio.gather(*tasks)

        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        
        survival_count = max(int(self.pop_size * 0.2), 4)
        top_performers = self.population[:survival_count]

        best_alpha = top_performers[0]
        if best_alpha["fitness"] > 0:
            HallOfFame.save_alpha(
                dna=best_alpha["dna"],
                metrics=best_alpha["metrics"],
                symbol=symbol
            )

        new_generation = copy.deepcopy(top_performers)
        while len(new_generation) < self.pop_size:
            parent_a, parent_b = random.sample(top_performers, 2)
            child = self._crossover(parent_a, parent_b)
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            new_generation.append(child)

        self.population = new_generation
        logger.info(f"🧬 Gen Complete. Top Alpha: {best_alpha['dna']} | Fitness: {best_alpha['fitness']:.4f}")

    def _crossover(self, p1, p2):
        split = random.randint(1, len(p1["dna"]) - 1)
        new_dna = p1["dna"][:split] + p2["dna"][split:]
        return {"dna": new_dna, "fitness": 0.0, "metrics": {}}

    def _mutate(self, individual):
        idx = random.randint(0, len(individual["dna"]) - 1)
        pools = [
            self.genes["momentum"], 
            self.genes["operators"], 
            self.genes["trend"], 
            self.genes["volatility"]
        ]
        individual["dna"][idx] = random.choice(pools[idx])
        return individual

    async def _calculate_fitness(self, individual, backtest_func, df):
        try:
            dna = individual["dna"]
            sig_momentum = DNALogic.resolve_expression(df, dna[0])
            operator     = dna[1]
            sig_trend    = DNALogic.resolve_expression(df, dna[2])
            sig_vol      = DNALogic.resolve_expression(df, dna[3])

            combined_base = DNALogic.apply_operator(operator, sig_momentum, sig_trend)
            final_signal = np.where(sig_vol > 0, combined_base, 0)

            config = {
                "dna_logic": dna,
                "custom_signal": final_signal,
                "is_genetic_run": True
            }

            result = await backtest_func(df_override=df, params=config)
            print(".", end="", flush=True)
  
            metrics = result.get("metrics", {})
            trades = metrics.get("totalTrades", 0)
            calmar = float(metrics.get("calmarRatio", 0.0))
            
            individual["metrics"] = metrics
            
            if trades < 5 or trades > 150:
                individual["fitness"] = calmar * 0.1
            else:
                individual["fitness"] = calmar
        except Exception as e:
            logger.error(f"Fitness Failure: {e}")
            individual["fitness"] = 0.0

            
# --- THE SOVEREIGN TRIGGER ---
if __name__ == "__main__":
    import argparse
    from app.backtest2 import execute_backtest

    parser = argparse.ArgumentParser(description="NEO-V7 Genetic Evolver")
    parser.add_argument("--symbol", type=str, default="BTC-USD")
    parser.add_argument("--gens", type=int, default=15)
    parser.add_argument("--pop", type=int, default=50)
    args = parser.parse_args()

    async def main():
        print(f"🧬 IGNITION: Starting Evolution for {args.symbol}")
        
        # 🟢 ABSOLUTE PATH FIX
        base_dir = os.getcwd()
        file_path = os.path.join(base_dir, "data", f"{args.symbol}-1h.csv")
        
        print(f"📂 Resolved Data Path: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df.set_index('datetime', inplace=True)
        except Exception as e:
            print(f"❌ DATA ERROR: Could not find {file_path}")
            return

        evolver = GeneticEvolver(pop_size=args.pop)
        for g in range(args.gens):
            print(f"\n--- 🧬 Generation {g} Processing ---")
            await evolver.evolve_generation(execute_backtest, df, symbol=args.symbol)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Nuclear Shutdown Initiated.")
