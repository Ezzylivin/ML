# NeoV6 — Server Recovery Runbook

The app is **3 pieces**. Only one lives on *your* server; the other two are managed
clouds that heal themselves.

| Piece | Where it runs | If it goes down |
|---|---|---|
| **Frontend** (React) | Vercel (managed) | Auto-recovers. Nothing to do. |
| **Backend** (Node/Express) | Render (managed) | Auto-recovers / auto-restarts. Nothing to do. |
| **Engine** (Python ML, this box) | `74.208.28.77:8000`, systemd `neo-engine.service` | **This runbook.** |

The engine is the only thing you ever have to touch. The fleet's open positions,
balances and child bots live in `bot_state.db` (SQLite in this directory), and the
engine restores them on startup when `AUTO_RESUME_BOTS=1`.

Health check (use after every recovery):
```
curl -s http://74.208.28.77:8000/api/health
```

---

## Tier 0 — "Is it actually down?"
```
curl -s http://74.208.28.77:8000/api/health     # from anywhere
systemctl status neo-engine.service              # on the box
```
If health returns JSON, the engine is up — the problem is elsewhere (check Vercel /
Render dashboards). If it hangs or refuses, continue below.

## Tier 1 — The server rebooted (power loss, host restart)
Nothing to do **if the service is enabled** — systemd starts it on boot and the fleet
auto-resumes. Confirm enablement once so you never get surprised:
```
systemctl is-enabled neo-engine.service          # want: "enabled"
sudo systemctl enable neo-engine.service          # run once if it says "disabled"
```
Then verify with the health check.

## Tier 2 — Engine crashed / stuck, box still up  (the usual case — ~5 seconds)
```
sudo systemctl restart neo-engine.service
curl -s http://74.208.28.77:8000/api/health
```
If it won't bind because a stray process still holds the port (the symptom is
"address already in use" in the logs), clear it first:
```
sudo fuser -k 8000/tcp
sudo systemctl restart neo-engine.service
```
Watch it come up:
```
journalctl -u neo-engine.service -n 50 -f
```

> ⚠️ Run the engine via **systemd only**. Do NOT start it with Docker or a manual
> `python main4.py` alongside systemd — a second process grabbing :8000 shadows the
> real one and serves stale code (this already bit us once). If `docker ps` shows a
> `neo-engine` container, `docker compose down` it.

## Tier 3 — Move the engine to a DIFFERENT server (migration)
The backend/frontend stay on Render/Vercel. MongoDB is external (same `MONGO_URI`),
so it follows automatically — **the only local data that must travel is the SQLite
state** (`bot_state.db`, `trading_state.db`, ledger, results). The only external
change is the engine's **IP**, which the Render backend must be re-pointed to.

Do it in this order to avoid downtime AND avoid two engines trading the same fleet.

### A. Prep on the OLD box (if it's still reachable)
```
bash backup_neov6.sh                       # fresh snapshot of all SQLite state
scp neov6-backup-<latest>.tgz  user@<NEW_IP>:/root/
scp /etc/systemd/system/neo-engine.service user@<NEW_IP>:/root/    # carries the env
```
If the old box is already dead, use your most recent off-box backup + your saved
copy of the unit file instead.

### B. Stand up the NEW box
```
sudo apt update && sudo apt install -y python3.12 python3.12-venv git
git clone <your ML repo URL> /root/Project/ML
cd /root/Project/ML
python3.12 -m venv /root/venv
/root/venv/bin/pip install -r requirements.txt
tar xzf /root/neov6-backup-<latest>.tgz -C /root/Project/ML     # restore state
sudo cp /root/neo-engine.service /etc/systemd/system/           # brings env vars
sudo systemctl daemon-reload
sudo systemctl enable --now neo-engine.service
curl -s http://<NEW_IP>:8000/api/health                         # expect JSON
```
Open port 8000 on the new box's firewall/security group (ideally restricted to
Render's egress, otherwise `0.0.0.0/0` to start).

If you didn't copy the unit file, recreate the env vars by hand — they are secrets,
not in git: `MONGO_URI`, `ALLOWED_ORIGINS`, `NODE_BACKEND_URL`, `INTERNAL_API_KEY`,
`ENGINE_API_KEY`, `AUTO_RESUME_BOTS=1`, plus the macro/self-learn/audit toggles.

### C. Cut over (this is the step people forget)
1. **Render → Environment:** set `ML_ENGINE_URL` (and `ML_SERVER_URL` if present) to
   `http://<NEW_IP>:8000`. Save → Render redeploys the backend. That single change
   is what makes the live app talk to the new engine.
2. `ALLOWED_ORIGINS` on the engine lists the **backend/frontend** origins, which did
   NOT change (Render/Vercel URLs are stable) — leave it. Only touch it if you also
   moved those domains.
3. **Stop the old engine so it can't double-trade the shared MongoDB fleet:**
   ```
   sudo systemctl disable --now neo-engine.service     # on the OLD box
   ```
4. End-to-end check: load the Vercel site, confirm the fleet roster + balances show
   and new activity ticks. `curl http://<NEW_IP>:8000/api/health` one more time.

> ⚠️ Never leave BOTH engines running against the same `MONGO_URI` — they'd each act
> on the same users and duplicate entries. New up → Render re-pointed → OLD stopped.

### Make the next migration trivial (do once)
Put a DNS name in front of the engine (e.g. `engine.yourdomain.com` → the box's IP)
and set Render's `ML_ENGINE_URL` to that hostname instead of a raw IP. Then a future
move is just a DNS record change — no Render edit, no redeploy.

---

## Make "down" rare (set up once)
- **Auto-start on boot:** `sudo systemctl enable neo-engine.service` (Tier 1).
- **Auto-restart on crash:** ensure the unit has, under `[Service]`,
  `Restart=always` and `RestartSec=5`, then `daemon-reload`. systemd then revives
  the engine within seconds of any crash — most Tier 2 events fix themselves.
- **Uptime alert:** add an UptimeRobot (or similar) HTTP monitor on
  `http://74.208.28.77:8000/api/health` so you hear about downtime before users do.
- **Backups:** schedule `backup_neov6.sh` via cron (daily) and copy the tarball
  OFF the box so Tier 3 is actually recoverable.
- **Keep off-box:** the `neo-engine.service` unit + its env values, and the repo URL.
