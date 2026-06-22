#!/usr/bin/env bash
# CoClaw WSL2 / Ubuntu environment setup (spec section 15).
#
# Run INSIDE WSL2 Ubuntu, from the project root on the Windows filesystem:
#   cd /mnt/c/Users/81309/OneDrive/Desktop/Idea/CoClaw
#   bash scripts/setup_wsl.sh
#
# Notes:
#  - The Python venv is created OUTSIDE the OneDrive-synced tree ($HOME/.venvs/coclaw)
#    to avoid OneDrive trying to sync thousands of venv files. Override with COCLAW_VENV.
#  - LKH-3 is built from source and copied to ~/.local/bin (added to PATH).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${COCLAW_VENV:-$HOME/.venvs/coclaw}"
BIN="$HOME/.local/bin"
LKH_BASE="http://akira.ruc.dk/~keld/research/LKH-3/"   # latest tarball auto-detected below

echo "== [1/4] apt dependencies =="
sudo apt-get update
sudo apt-get install -y build-essential wget python3-venv python3-pip

echo "== [2/4] python venv at $VENV =="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
pip install -r "$PROJ/requirements.txt"

echo "== [3/4] LKH-3 (main reference solver) =="
mkdir -p "$BIN"
if command -v LKH >/dev/null 2>&1 || [ -x "$BIN/LKH" ]; then
  echo "LKH already present, skipping build."
else
  tmp="$(mktemp -d)"
  (
    cd "$tmp"
    tgz="$(curl -s "$LKH_BASE" | grep -oE 'LKH-3\.0\.[0-9]+\.tgz' | sort -uV | tail -1)"
    [ -n "$tgz" ] || tgz="LKH-3.0.14.tgz"
    echo "downloading $tgz"
    wget -q "${LKH_BASE}${tgz}"
    tar xf "$tgz"
    cd "${tgz%.tgz}"
    make
    cp LKH "$BIN/"
  )
  rm -rf "$tmp"
fi
if ! echo ":$PATH:" | grep -q ":$BIN:"; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  export PATH="$BIN:$PATH"
  echo "added ~/.local/bin to PATH in ~/.bashrc"
fi
echo "LKH -> $(command -v LKH || echo "$BIN/LKH")"

echo "== [4/4] done =="
cat <<EOF

Next:
  source $VENV/bin/activate          # activate the venv in new shells
  export DEEPSEEK_API_KEY=...         # or: cp .env.example .env && edit, then  set -a && source .env && set +a
  python scripts/sanity_check_lcar.py # should PASS (no solver/LLM needed)
  python -m experiments.gates --config configs/default.yaml   # once gates.py is implemented

Optional later:
  pip install gurobipy               # + grbgetkey <academic-license-key>   (gate 3 exact check)
  pip install sentence-transformers  # embedding-based skill retrieval (else return-all fallback)
EOF
