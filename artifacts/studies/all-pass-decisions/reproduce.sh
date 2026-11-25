#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
python code/render_decision_study.py --inputs inputs --output generated
python code/check_estimates.py
python code/verify_selector_inputs.py
if [ -e inputs/external_groups.csv ]; then
    python code/external_summary.py
fi
if [ "${1:-}" = "--pdf" ]; then
    if [ -f paper.tex ]; then tectonic paper.tex; else tectonic exhibits.tex; fi
fi
