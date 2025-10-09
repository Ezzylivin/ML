import os
from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import json

# Initialize the Flask application
app = Flask(__name__)

# This is crucial to allow your React frontend to talk to this API
CORS(app)

# --- Configuration ---
TRADE_LOG_FILE = 'paper_trades.csv'
STATUS_FILE = 'bot_status.json'

# --- API Endpoints ---

@app.route('/trades')
def get_trades():
    """Returns all trades from the paper trading log."""
    if not os.path.exists(TRADE_LOG_FILE):
        return jsonify([]) # Return empty list if no trades yet
    try:
        df = pd.read_csv(TRADE_LOG_FILE)
        trades = df.to_dict(orient='records')
        return jsonify(trades)
    except Exception as e:
        return jsonify({"error": f"Failed to process trade log: {e}"}), 500

@app.route('/status')
def get_status():
    """Reads the status file written by the bot and returns its content."""
    if not os.path.exists(STATUS_FILE):
        # If the bot hasn't run yet, return a default 'stopped' status
        return jsonify({"status": "stopped", "isConfigured": False})
    try:
        with open(STATUS_FILE, 'r') as f:
            status_data = json.load(f)
        return jsonify(status_data)
    except Exception as e:
        return jsonify({"error": f"Failed to read status file: {e}"}), 500

# This block allows us to run the script directly
if __name__ == '__main__':
    app.run(debug=True, port=5001)
