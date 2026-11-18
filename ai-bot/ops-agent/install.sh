#!/bin/sh
# Installs the ai-bot ops agent on this host and lets ai-bot's ops key run it, and nothing else.
#   sh install.sh 'ssh-ed25519 AAAA... ai-bot-ops'
# Run as the user the bot logs in as (ubuntu), from this folder; needs sudo. Safe to re-run (replaces the old key line).
set -eu
KEY="${1:-}"
case "$KEY" in
  ssh-ed25519\ *ai-bot-ops) ;;
  *) echo "usage: sh install.sh 'ssh-ed25519 AAAA... ai-bot-ops'" >&2; exit 1 ;;
esac
sudo install -m 755 -o root -g root "$(dirname "$0")/ai-bot-ops" /usr/local/bin/ai-bot-ops
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
{ grep -v ' ai-bot-ops$' ~/.ssh/authorized_keys || true; echo "command=\"/usr/local/bin/ai-bot-ops\",restrict $KEY"; } > ~/.ssh/authorized_keys.new
chmod 600 ~/.ssh/authorized_keys.new
mv ~/.ssh/authorized_keys.new ~/.ssh/authorized_keys
echo "ai-bot-ops installed on $(hostname)"
