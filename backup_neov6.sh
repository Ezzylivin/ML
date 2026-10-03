#!/usr/bin/env bash
# backup_neov6.sh — snapshot the engine's critical, non-regenerable state.
# Backs up bot state, the trade ledger, and the validation results into a
# timestamped tarball, keeps the last 14 days, and (optionally) pushes offsite.
#
# Schedule it (every 6h) with:
#   crontab -e   then add:
#   0 */6 * * * /root/Project/ML/backup_neov6.sh >> /root/Project/ML/logs/backup.log 2>&1
#
# NOTE: this does NOT back up .env (it holds secrets). Store that separately/securely.
set -euo pipefail

APP_DIR="/root/Project/ML"
DEST="$APP_DIR/backups"
TS="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST" "$APP_DIR/logs"
cd "$APP_DIR"

# Collect whatever state files exist (guarded so a missing one doesn't abort).
FILES=()
for f in bot_state.db trading_state.db exit_lab_results.json; do
  [ -f "$f" ] && FILES+=("$f")
done
# Any databases / result json under data/
if [ -d data ]; then
  while IFS= read -r -d '' p; do FILES+=("$p"); done \
    < <(find data -maxdepth 2 -type f \( -name '*.db' -o -name '*_results.json' \) -print0 2>/dev/null)
fi

if [ ${#FILES[@]} -eq 0 ]; then
  echo "[$(date -Is)] backup: no state files found — nothing to do"; exit 0
fi

ARCHIVE="$DEST/neov6-state-$TS.tar.gz"
tar -czf "$ARCHIVE" "${FILES[@]}"
echo "[$(date -Is)] backup: wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1), ${#FILES[@]} files)"

# Prune backups older than 14 days.
find "$DEST" -name 'neov6-state-*.tar.gz' -mtime +14 -delete 2>/dev/null || true

# --- OPTIONAL offsite copy (uncomment + configure ONE) ---
# rclone copy "$ARCHIVE" remote:neov6-backups/        # needs `rclone config` once
# aws s3 cp "$ARCHIVE" s3://your-bucket/neov6-backups/ # needs awscli + creds
# scp "$ARCHIVE" user@backup-host:/path/neov6-backups/ # needs ssh key
