#!/usr/bin/env bash
cd "$(dirname "$0")"
export STEMNOTES_HOME="$(pwd)"
exec venv/bin/python -m stemnotes.updater "$@"
