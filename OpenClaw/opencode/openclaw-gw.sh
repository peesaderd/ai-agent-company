#!/bin/bash
export PATH="/home/openhands/.npm-global/bin:$PATH"
if [ -f /home/openhands/openclaw/.env ]; then
  set -a
  source /home/openhands/openclaw/.env
  set +a
fi
# Use OPENCODE_API_KEY so opencode-go provider auto-detects
export OPENCODE_API_KEY="sk-LTP2Z9x9adJjxgzUfcWjoQS9lxekHw5xMhKUs5NkCCULT9jhCryWgCFOPdwfngi0"
export OPENAI_API_KEY="sk-LTP2Z9x9adJjxgzUfcWjoQS9lxekHw5xMhKUs5NkCCULT9jhCryWgCFOPdwfngi0"
# No OPENAI_BASE_URL - let opencode-go provider use its built-in endpoint
exec openclaw gateway run --port 18789 --bind lan --force
