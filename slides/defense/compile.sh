#!/bin/bash
# Convenience wrapper: build only this deck.
set -euo pipefail
exec "$(cd "$(dirname "$0")/.." && pwd)/build_slides.sh" defense
