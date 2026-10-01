#!/usr/bin/env bash
# Backs up the bot state + trade ledger with a timestamp, keeps last 30 copies.
set -euo pipefail
PROJECT=/root/Project/ML
DEST="$PROJECT/backups"
mkdir -p "$DEST"
STAMP=$(date -u +%Y%m%d-%H%M%S)
for db in bot_state.db data/trade_ledger.db trading_state.db; do
  if [ -f "$PROJECT/$db" ]; then
    base=$(basename "$db")
    # sqlite-consistent copy (works while the DB is in use)
    sqlite3 "$PROJECT/$db" ".backup '$DEST/${base%.db}-$STAMP.db'" 2>/dev/null \
      || cp "$PROJECT/$db" "$DEST/${base%.db}-$STAMP.db"
  fi
done
# prune: keep newest 30 of each prefix
for prefix in bot_state trade_ledger trading_state; do
  ls -1t "$DEST/$prefix"-*.db 2>/dev/null | tail -n +31 | xargs -r rm -f
done
echo "$(date -u +%FT%TZ) backup ok -> $DEST"
