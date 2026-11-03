#!/usr/bin/env python3
"""jpp_cli.py with its WikiText pool replaced by the JSON list at $PROMPTS_JSON.

fit-shard has --prompts-json but fit-router streams WikiText; this keeps the upstream clone
unmodified. Run from jlens_jpp_reproduction/jpp_lens/. The stamps do not record the pool: keep the JSON beside the fit.
"""
import os
import sys

sys.path.insert(0, "scripts")
import jpp_cli

if __name__ == "__main__":
    pool = os.environ["PROMPTS_JSON"]
    jpp_cli.load_fit_prompts = lambda n: jpp_cli.load_fit_prompts_from_json(pool, n)
    raise SystemExit(jpp_cli.main())
