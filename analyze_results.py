import optuna
import os
import pandas as pd

# Path to your results
RESULTS_DIR = "/root/Project/ML/data/optimizer_results"

def analyze_latest_study():
    # 1. Find the latest database file
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".db")]
    if not files:
        print("❌ No study databases found!")
        return

    # Sort by modified time to get the one you just ran
    latest_db = max([os.path.join(RESULTS_DIR, f) for f in files], key=os.path.getmtime)
    study_name = os.path.basename(latest_db).replace(".db", "")
    storage_url = f"sqlite:///{latest_db}"

    print(f"🔍 Analyzing Study: {study_name}")
    print(f"📂 Source: {latest_db}")

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        print(f"\n📊 Total Trials Run: {len(study.trials)}")
        
        # Filter for completed trials
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"✅ Completed Trials: {len(completed)}")
        
        if not completed:
            print("⚠️ No trials completed successfully. Check error logs.")
            return

        # Sort by value (Calmar Ratio) descending
        sorted_trials = sorted(completed, key=lambda t: t.values[0], reverse=True)
        
        print("\n🏆 TOP 5 RESULTS (Even if they failed certification):")
        print("-" * 60)
        for i, t in enumerate(sorted_trials[:5]):
            calmar = t.values[0]
            dd = t.values[1]
            print(f"Rank #{i+1} | Calmar: {calmar:.4f} | Max DD: {dd:.2f}%")
            print(f"   👉 Strategy: {t.user_attrs.get('combo_strategies', 'Unknown')}")
            print(f"   👉 Params: {t.params}")
            print("-" * 60)

        best = study.best_trials[0]
        print(f"\n🥇 ABSOLUTE BEST PARAMS FOUND:")
        print(best.params)

    except Exception as e:
        print(f"❌ Error loading study: {e}")

if __name__ == "__main__":
    analyze_latest_study()
