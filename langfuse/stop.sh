#!/bin/bash
# Stop Langfuse (data is preserved in Docker volume)
cd "$(dirname "$0")"
docker compose down
echo "  Langfuse stopped. Data is preserved — run start.sh to resume."
