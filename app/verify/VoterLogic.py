import numpy as np

def calculate_consensus_signal(strategy_signals, threshold=0.5):
    """
    strategy_signals: List of signals [-1 (Short), 0 (Neutral), 1 (Long)]
    threshold: % of agreement required (e.g., 0.5 = 50% majority)
    """
    signal_array = np.array(strategy_signals)
    n_strategies = len(signal_array)
    
    # Calculate sum of votes
    total_longs = np.sum(signal_array == 1)
    total_shorts = np.sum(signal_array == -1)
    
    votes_required = int(n_strategies * threshold)
    
    print(f"--- 🗳️ Voting Session ---")
    print(f"Strategies: {n_strategies} | Threshold: {votes_required} votes")
    print(f"Long Votes: {total_longs} | Short Votes: {total_shorts}")

    if total_longs >= votes_required:
        print("✅ CONSENSUS: STRONG BUY")
        return 1
    elif total_shorts >= votes_required:
        print("✅ CONSENSUS: STRONG SELL")
        return -1
    else:
        print("❌ NO CONSENSUS: STANDING ASIDE")
        return 0

# Example Usage for your 10-strat pool:
# signals = [1, 1, 0, 1, -1, 1, 0, 0, 1, 1] (6 Longs out of 10)
# calculate_consensus_signal(signals, threshold=0.6)
