# NEO — Security Hardening Checklist (Phase 3)

Do these in order. Items marked ⚠️ touch code I couldn't see (botController) or
infra I can't run — apply them yourself and test bot control after each.

## 1. Lock down the engine's port 8000 (highest priority)
Right now `main4.py` binds `0.0.0.0:8000` over plain HTTP with no auth — anyone
who finds the IP can drive your bot. Pick ONE:

### Option A (recommended): shared-secret header on the engine ⚠️
The engine only answers requests carrying a secret header that your Node backend
adds. Works even though Render's egress IPs are dynamic. You already use this
pattern (INTERNAL_API_KEY in socket_emitter).

1. On the engine, add a FastAPI dependency that checks `X-Internal-Key` against
   `os.getenv("ENGINE_API_KEY")` on every `/api/*` route.
2. In the Node backend, add `headers: { "X-Internal-Key": process.env.ENGINE_API_KEY }`
   to EVERY call it makes to the engine — botController, ledgerRoutes, etc.
3. Set the same `ENGINE_API_KEY` (a long random string) on both hosts.
Enable only after step 2 is done everywhere, or you'll lock out your own bot
control. Test start/stop/status immediately after.

### Option B: firewall to known sources
If your Node backend has a static outbound IP, restrict 8000 to it:
    ufw allow from <NODE_EGRESS_IP> to any port 8000
    ufw deny 8000
    ufw enable
(Render's shared plans don't give static egress IPs, so Option A is usually the
real answer.)

## 2. Exchange API keys
- Confirm keys are **trade-only, withdrawals disabled** on Coinbase/Kraken. This
  is your biggest blast-radius control: even a full compromise can't withdraw.
- Keys currently live only in memory (ACTIVE_BOTS) and are redacted before DB
  writes (_redact_secrets) — good. Never add them to logs or the ledger.
- If you ever persist them, encrypt at rest (e.g. libsodium/Fernet with a key
  from the environment, not in the repo).

## 3. Secrets management
- Ensure `.env` is in `.gitignore` and was never committed (check history:
  `git log --all -p -- .env`). If it was, rotate every secret in it now.
- Rotate MONGO_URI, any exchange keys, JWT secret, and INTERNAL_API_KEY.
- Prefer host env vars / a secrets manager over a committed file.

## 4. Per-user isolation
- The ledger now supports per-user scoping (summary(user_id=...)), and
  ledgerRoutes derives the id from the verified token. Confirm `botUserId(req)`
  returns the SAME id the bot is started with (the wallet address) — otherwise a
  user sees an empty ledger. Test with two accounts before going multi-user.

## 5. Transport
- Frontend → Node backend is HTTPS ✅. Node → engine is internal HTTP; that's
  acceptable once #1 (shared secret) is in place. If the engine ever serves
  browsers directly, put it behind HTTPS (Caddy/Nginx + domain).

## 6. Dependencies & surface
- `pip list --outdated` and `npm audit` on both backends; patch criticals.
- helmet + rate limiting are on the Node app ✅. Add rate limiting to the engine
  if it's ever exposed beyond the Node backend.

## 7. Operational safety
- Keep `maxDailyLoss` / `maxDrawdown` conservative in live.
- Verify the kill path: `POST /api/bot/stop` halts a bot. For a true emergency
  flatten of a LIVE position, close it on the exchange directly — don't rely on
  automated flattening you haven't tested.
