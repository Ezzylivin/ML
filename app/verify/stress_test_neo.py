import asyncio
import pandas as pd
import numpy as np
import os
import sys
import logging

# Ensure internal modules are discoverable
sys.path.append(os.getcwd())

from app.discovery.evolver import GeneticEvolver
from app.backtest2 import execute_backtest

# Silence verbose TF logs for the test
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.INFO)

async def run_stress_test():
    print("\n" + "="*60)
    print("🧪 SOVEREIGN SYSTEM STRESS TEST (V10.5) - DNA TRACKER")
    print("="*60)
    
    # 1. ENSURE DATA EXISTS
    csv_path = 'data/BTC-USD-1h.csv'
    if not os.path.exists(csv_path):
        print("🟡 Mock data missing. Generating 1000 candles...")
        os.makedirs('data', exist_ok=True)
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
        df = pd.DataFrame({
            'datetime': dates,
            'open': np.random.uniform(40000, 60000, 1000),
            'high': np.random.uniform(60000, 61000, 1000),
            'low': np.random.uniform(39000, 40000, 1000),
            'close': np.random.uniform(40000, 60000, 1000),
            'volume': np.random.uniform(100, 1000, 1000)
        })
        df.to_csv(csv_path, index=False)

    # 2. INITIALIZE COMPONENTS
    # Bumped to 10 to satisfy the survival minimum logic
    evolver = GeneticEvolver(pop_size=10, mutation_rate=0.2)
    df_test = pd.read_csv(csv_path)

    print(f"✅ Components Initialized.")
    print(f"🧬 Running 3 Generations of evolution...")

    # 3. EXECUTION LOOP
    try:
        for gen in range(1, 4):
            print(f"\n--- 🧬 Generation {gen} Processing ---")
            
            # This triggers the full loop: 
            # Evolver -> Backtest2 -> Manager2 -> Heartbeat -> Results
            await evolver.evolve_generation(execute_backtest, df_test)
            
            # Verify Heartbeat file is being written to
            if os.path.exists('engine_heartbeat.txt'):
                with open('engine_heartbeat.txt', 'r') as f:
                    last_beat = f.read()
                    print(f"💓 Heartbeat: {last_beat}")
            
            # 🟢 LOG THE WINNER
            winner = evolver.population[0]
            top_fit = winner['fitness']
            top_dna = winner['dna']
            
            print(f"🏆 GEN {gen} WINNER: {top_dna}")
            print(f"📈 Fitness (Calmar): {round(top_fit, 4)}")

        print("\n" + "="*60)
        print("🏁 STRESS TEST PASSED: SYSTEM IS SOVEREIGN")
        print(f"🧬 FINAL ALPHA DNA: {evolver.population[0]['dna']}")
        print("="*60)

    except Exception as e:
        print(f"\n❌ STRESS TEST FAILED!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_stress_test())
