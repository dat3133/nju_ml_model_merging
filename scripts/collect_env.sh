#!/usr/bin/env bash
set -euo pipefail

{
  echo "# Environment"
  echo
  echo "## Date"
  date -u
  echo
  echo "## Python"
  python --version
  echo
  echo "## CUDA / GPU"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi not found"
  fi
  echo
  echo "## pip freeze"
  pip freeze
} > environment_report.md

echo "wrote environment_report.md"
