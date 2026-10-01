import time
import os

def watch_progress():
    print("🔭 Watching ML Engine Progress...")
    # This assumes we add a line to backtest2.py to write to this file
    log_file = "engine_heartbeat.txt"
    
    if not os.path.exists(log_file):
        with open(log_file, "w") as f: f.write("Starting...")

    last_stat = ""
    while True:
        try:
            with open(log_file, "r") as f:
                current = f.read().strip()
                if current != last_stat:
                    print(f"📡 ML SIGNAL: {current}")
                    last_stat = current
        except: pass
        time.sleep(1)

if __name__ == "__main__":
    watch_progress()
