#!/bin/bash
# Start Langfuse locally — runs at http://localhost:3000
cd "$(dirname "$0")"
docker compose up -d
echo ""
echo "  Langfuse starting at http://localhost:3000"
echo "  First run takes ~60 seconds to download images."
echo "  To stop:  ./stop.sh"
