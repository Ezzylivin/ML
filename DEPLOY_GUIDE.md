# NEO — Phase 1–4 Deployment Guide

main4.py changes are ALREADY on the server (state recovery, /api/health, ledger
user_id). Below is everything else. Do the engine/server pieces first, then the
Node backend, then the frontend.

## A. Copy files to the server (from your PC, PowerShell)
    scp -i <key> ledger_report.py            root@74.208.28.77:/root/Project/ML/
    scp -i <key> walkforward_backtest.py     root@74.208.28.77:/root/Project/ML/
    scp -i <key> trade_recorder.py           root@74.208.28.77:/root/Project/ML/app/services/
    scp -i <key> backup_state.sh             root@74.208.28.77:/root/Project/ML/
    scp -i <key> neo-engine.service          root@74.208.28.77:/root/
    scp -i <key> neo-backup.service          root@74.208.28.77:/root/
    scp -i <key> neo-backup.timer            root@74.208.28.77:/root/

## B. On the server: run the engine + backups under systemd
    # stop the manual tmux engine first
    pkill -f main4.py; tmux kill-session -t neo 2>/dev/null
    mkdir -p /root/Project/ML/logs
    mv /root/neo-engine.service /root/neo-backup.service /root/neo-backup.timer /etc/systemd/system/
    chmod +x /root/Project/ML/backup_state.sh
    systemctl daemon-reload
    systemctl enable --now neo-engine.service     # engine now auto-restarts on crash/reboot
    systemctl enable --now neo-backup.timer
    systemctl status neo-engine.service --no-pager | head -15
    # (the retrainer timer from before, neo-retrain.timer, stays as-is)

## C. Verify the engine
    curl -s http://localhost:8000/api/health | python3 -m json.tool
    # -> status ok, active_bots, running_tasks, ledger stats

## D. Edge-validation tools (run these over time — this is Phase 1's real answer)
    cd /root/Project/ML
    /root/venv/bin/python ledger_report.py                 # performance from real paper trades
    /root/venv/bin/python walkforward_backtest.py --folds 6  # out-of-sample model edge

## E. Node backend (your repo -> Render)
    - Update routes/ledgerRoutes.js (now scopes by authenticated user).
    - Confirm ML_ENGINE_URL matches how botController reaches the engine.
    - Push -> Render redeploys.

## F. Frontend (your repo -> Vercel)
    - Add src/pages/LedgerDashboard.jsx, the /dashboard/ledger route, and the
      internal nav <Link> (from earlier).
    - Push -> Vercel redeploys.

## G. Security — follow SECURITY.md
    Priority #1: shared-secret header (or firewall) on engine :8000. Do this
    before any real money is live.

## Notes
- State recovery: restarting a bot now RESUMES open positions (paper-test it).
  Use POST /api/bot/reset for a clean slate.
- systemd runs the engine as one process; do NOT also run it in tmux.
