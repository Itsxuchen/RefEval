#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
python code/render_decision_study.py --inputs inputs --output generated
python code/cross_scorer_exhibits.py
python code/full_audit_exhibits.py
python code/check_estimates.py
python code/verify_selector_inputs.py
if [ -e inputs/external_groups.csv ]; then
    python code/external_summary.py
fi
if [ "${1:-}" = "--pdf" ]; then
    tectonic exhibits.tex
    if [ -f paper.tex ]; then tectonic paper.tex; fi
fi
