#!/usr/bin/env bash
# Synthetic end-to-end smoke test -- CPU only, no real data, about 10-15 minutes.
#
#   bash tests/smoke_test.sh                     # work folder tests/_work/ (git-ignored), removed first
#   bash tests/smoke_test.sh --work /tmp/ctfm    # another work folder
#   bash tests/smoke_test.sh --random-encoder    # allow a randomly initialised CT-FM encoder (see below)
#   bash tests/smoke_test.sh --with-few-shot     # also the few-shot regime (Radcure OS, sizes 10/24/50): +10 min
#
# What runs (the pipeline's own scripts, through runner.py as run_pipeline.sh calls them):
#   1  tests/make_synthetic_data.py            fake Radcure (120) / Maastro (80) cohorts: tables + 160x160x64 NIfTI scans
#   2  data_preparation/build_*_corrected.py   corrected tables (pure LR / DM events), with their validations
#   3  runner.py features --which st    CT-FM features, the paper's CT-FM-ST input, on the CPU
#   4  run_a1.py                               CT-FM-ST + Clinical only + Clinical + CT-FM-ST: OS (original data) and
#                                              LR (corrected data), full shot, both cohorts (+ few shot: --with-few-shot)
#   5  run_external.py --endpoint os           external validation Radcure -> Maastro
#   5b run_spectre_cox.py                      SPECTRE + Cox head (OS) -- the CT-FM features stand in for the SPECTRE
#                                              features (the SPECTRE model itself is not run here)
#   6  make_table_s4.py, make_km.py            Table S4 (C-index with 95% CI) and the Kaplan-Meier figures
# Test-only settings (config "testing" section, written to the work folder's config_test.yaml):
#   bootstrap_b = 20             bootstrap resamples instead of the pipeline's 2,000
#   random_init_encoder          only with --random-encoder, and only if the CT-FM weights
#                                (HuggingFace project-lighter/ct_fm_feature_extractor) cannot be loaded -- e.g. offline
#                                without a local copy. Its features are meaningless; the code path is the same.
# Synthetic data and small test sets: the numbers mean nothing. The test checks that every step runs and writes
# its outputs.
set -euo pipefail
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
WORK="$REPO/tests/_work"; RANDOM_OK=0; FEW=0; B=20
while [ $# -gt 0 ]; do
  case "$1" in
    --work) WORK="$2"; shift 2;;
    --random-encoder) RANDOM_OK=1; shift;;
    --with-few-shot) FEW=1; shift;;
    *) echo "unknown option: $1"; exit 2;;
  esac
done
PY=${PY:-python3}
PY=$(command -v "$PY") || { echo "python interpreter not found (set PY)"; exit 2; }
export CUDA_VISIBLE_DEVICES=""                  # CPU only
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
for v in $(env | sed -n 's/^\(CTFM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
T0=$(date +%s)
say() { echo; echo "[smoke $(( $(date +%s) - T0 ))s] $*"; }
fail() { echo; echo "SMOKE TEST FAILED: $*"; exit 1; }

case "$WORK" in /|"$HOME"|"$REPO") fail "refusing to use $WORK as the work folder";; esac
if [ -d "$WORK" ] && [ -n "$(ls -A "$WORK")" ] && [ ! -f "$WORK/config_test.yaml" ]; then
  fail "$WORK is not empty and is not a smoke-test folder -- choose another --work"
fi
rm -rf "$WORK"; mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
cat > "$WORK/config_test.yaml" <<EOF
# smoke-test configuration (tests/smoke_test.sh); relative paths are relative to this file
data:
  radcure_csv: data/new_radcure_data.csv
  maastro_csv: data/new_maastro_data.csv
  radcure_csv_corrected: data/new_radcure_data_corrected_main.csv
  maastro_csv_corrected: data/new_maastro_data_corrected_main.csv
  maastro_pure_events_csv: data/maastro_pure_lr_dm_events.csv
  radcure_images: data/imagesTs_radcure
  maastro_images: data/imagesTs_maastro
  radcure_masks: data/predsTs_radcure
  maastro_masks: data/predsTx_maastro
cache: {dir: cache}
outputs: {dir: outputs, final_dir: outputs/figures_and_tables, paper_run: null}
runtime: {python: "$PY", gpus: "0", spectre_gpu: 0, device: cpu}
testing: {random_init_encoder: false, bootstrap_b: $B}
EOF
export CTFM_CONFIG="$WORK/config_test.yaml"
cd "$REPO"
R="$PY $REPO/pipeline/runner.py"

say "1  synthetic cohorts"
"$PY" tests/make_synthetic_data.py --out "$WORK/data" --n-radcure 120 --n-maastro 80 --seed 0

say "2  corrected tables (data_preparation)"
"$PY" data_preparation/build_radcure_corrected.py > "$WORK/build_radcure.log" || { cat "$WORK/build_radcure.log"; fail "build_radcure_corrected.py"; }
"$PY" data_preparation/build_maastro_corrected.py > "$WORK/build_maastro.log" || { cat "$WORK/build_maastro.log"; fail "build_maastro_corrected.py"; }
grep -c PASS "$WORK"/build_*.log | sed 's/^/   validations passed: /'
"$PY" pipeline/ctfm_config.py check | grep -E 'data\.|cache|outputs\.dir'

say "3  CT-FM encoder and features (CPU)"
WEIGHTS=$("$PY" pipeline/ctfm_config.py get model.ctfm_weights)
if "$PY" -c "from huggingface_hub import try_to_load_from_cache as t; import sys; sys.exit(0 if isinstance(t(sys.argv[1], 'config.json'), str) else 1)" "$WEIGHTS" 2>/dev/null; then
  export HF_HUB_OFFLINE=1                       # in the local HuggingFace cache: no network needed
fi
if "$PY" -c "import sys; sys.path.insert(0, 'pipeline'); from ctfm_config import ctfm_encoder; ctfm_encoder()" > "$WORK/encoder.log" 2>&1; then
  echo "   CT-FM weights: $WEIGHTS (pretrained)"
elif [ "$RANDOM_OK" = 1 ]; then
  export CTFM_TESTING_RANDOM_INIT_ENCODER=true
  echo "   !! CT-FM weights could not be loaded (see $WORK/encoder.log): RANDOMLY INITIALISED encoder (--random-encoder)"
else
  tail -3 "$WORK/encoder.log"
  fail "the CT-FM weights ($WEIGHTS) could not be loaded. Connect to the internet once, set model.ctfm_weights to a local copy, or re-run with --random-encoder"
fi
$R features --which st > "$WORK/features.log" 2>&1 || { tail -20 "$WORK/features.log"; fail "CT-FM features"; }
grep saved "$WORK/features.log" | sed 's/^/  /'

RUN=20000101_000000
ROOT="$WORK/outputs/${RUN}_study"; mkdir -p "$ROOT/logs"
job() { local name=$1; shift; "$@" > "$ROOT/logs/$name.log" 2>&1 || { tail -25 "$ROOT/logs/$name.log"; fail "$name"; }; echo "   ok  $name"; }

say "4  CT-FM-ST + Clinical only + Clinical + CT-FM-ST (run_a1.py, $B bootstrap resamples)"
for co in Radcure Maastro; do
  job a1_original_${co}_os $R run --data original --features st --root "$ROOT" -- run_a1.py --run $RUN --cohort $co --endpoint os --regime full
  job a1_corrected_${co}_lr $R run --data corrected --features st --root "$ROOT" -- run_a1.py --run $RUN --cohort $co --endpoint lr --regime full
done
if [ "$FEW" = 1 ]; then
  job a1_original_Radcure_os_few $R run --data original --features st --root "$ROOT" -- run_a1.py --run $RUN --cohort Radcure --endpoint os --regime few
fi

say "5  external validation Radcure -> Maastro (OS)"
job external_original $R run --data original --features st --root "$ROOT" -- run_external.py --run $RUN --spec primary --endpoint os

say "5b SPECTRE + Cox head (OS), with the CT-FM features as stand-in SPECTRE features"
mkdir -p "$WORK/cache/features_spectre_paper" && cp "$WORK/cache/features_paper_st/"*.npz "$WORK/cache/features_spectre_paper/"
SC="$PY $REPO/pipeline/run_spectre_cox.py"
for co in Radcure Maastro; do job spectre_cox_$co $SC a1 --cohort $co --root "$ROOT/spectre_cox" --run $RUN --regime full; done
job spectre_cox_external $SC external --root "$ROOT/spectre_cox" --run $RUN
job spectre_cox_rows $SC rows --root "$ROOT/spectre_cox" --main "$ROOT" --out "$ROOT/report_original"
[ -f "$ROOT/report_original/spectre_cox_rows.md" ] || fail "SPECTRE + Cox head rows not written"

say "6  Table S4 and Kaplan-Meier figures"
A1="$ROOT/results_original_n8_$RUN"
job table_s4_original $R run --data original --root "$ROOT" -- make_table_s4.py --runs "$A1" --out "$ROOT/report_original" --procs 4
job km_original $R run --data original --root "$ROOT" -- make_km.py --runs "$A1" --out "$ROOT/report_original/figures"

say "checks"
"$PY" - "$ROOT" "$FEW" <<'EOF'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
root = Path(sys.argv[1]); run = root.name.replace("_study", "")
A1, EXT, REP = root / f"results_original_n8_{run}", root / f"results_external_original_n8_{run}", root / "report_original"
need = [REP / "table_s4_cindex.md", REP / "table_s4_cindex_long.csv", REP / "figures" / "fig_km_tertiles_radcure.png",
        A1 / "run_info.json", A1 / "_cells" / "rows_full_Radcure_os_primary.csv",
        root / f"results_corrected_n8_{run}" / "_cells" / "rows_full_Maastro_lr_primary.csv",
        EXT / f"external_results_Radcure_to_Maastro_refit_train_primary.csv"]
missing = [str(p) for p in need if not p.exists()]
assert not missing, f"missing outputs: {missing}"
t = pd.read_csv(REP / "table_s4_cindex_long.csv", dtype={"train_size": str})
full = t[(t.train_size == "full") & (t.endpoint == "OS")]
models = {"CT-FM-ST", "Clinical only", "Clinical + CT-FM"}
for co in ("Radcure", "Maastro"):
    got = set(full[full.cohort == co].model)
    assert models <= got, f"{co}: Table S4 lacks {models - got}"
assert np.isfinite(full[["value", "lo", "hi"]].to_numpy()).all(), "non-finite C-index / CI"
assert (full.lo <= full.value).all() and (full.value <= full.hi).all(), "point estimate outside its CI"
if sys.argv[2] == "1":
    few = t[(t.cohort == "Radcure") & (t.train_size != "full")]
    assert len(few) and set(few.train_size) >= {"10", "24", "50"}, "few-shot rows missing"
e = pd.read_csv(EXT / "external_results_Radcure_to_Maastro_refit_train_primary.csv")
assert np.isfinite(e.cindex).all() and len(e) == 3, "external results incomplete"
print(f"   Table S4: {len(t)} cells ({t.model.nunique()} models); full-shot OS C-index:")
for r in full.sort_values(["cohort", "model"]).itertuples():
    print(f"      {r.cohort:8} {r.model:18} {r.value:.3f} ({r.lo:.3f}-{r.hi:.3f})  test n={r.n_test}, {r.n_events} events")
print(f"   external OS (Radcure -> all {int(e.n_target.iloc[0])} Maastro): "
      + ", ".join(f"{a} {c:.3f}" for a, c in zip(e.arm, e.cindex)))
EOF
say "SMOKE TEST PASSED (synthetic data: the numbers are meaningless)"
echo "   outputs: $ROOT"
echo "   table:   $ROOT/report_original/table_s4_cindex.md"
