#!/usr/bin/env bash
# The submitted paper's pipeline, re-run with 95% CIs (see runner.py for what "the paper's pipeline" is and
# how it was recovered). Internal (full + few shot, 5 seeds, all six models), subgroups, external validation
# (CT-FM-ST, clinical only, Clinical + CT-FM-ST on all 173 Maastro), tables and figures.
#   OS, RFS : the submitted data (reproduces the paper)      LR, DM : corrected (pure) LR / DM events
#
#   bash run_pipeline.sh --gpus "0 2 3" --spectre-gpu 3         # launch (tmux session ctfm_<run>)
#   bash run_pipeline.sh --gpus "0 2 3" --dry-run               # show the plan and check inputs; launch nothing
#   bash run_pipeline.sh --run <run id> --gpus "0 2 3"          # resume a stopped run (finished steps are kept)
#
# Locations come from config.yaml (or the file in CTFM_CONFIG) plus CTFM_* environment overrides -- see
# pipeline/ctfm_config.py. --gpus / --spectre-gpu default to runtime.gpus / runtime.spectre_gpu; the Python
# interpreter is $PY, else runtime.python. The resolved configuration is written to the run folder
# (config_resolved.yaml) and every job of the run reads that file.
#
# Steps (main window; SegRes-E2E workers in windows gpu<N>):
#   0  paper-input CT-FM features st, lp, lp2 (GPU, first of --gpus), SPECTRE features (--spectre-gpu), SegRes scan caches (CPU)
#   1  verification against the paper's saved models and per-seed results (report/verify.md; never stops the run)
#   2  SegRes-E2E: 300 networks on --gpus (fp32)   |  in parallel on CPU: CT-FM-ST + clinical + combined (internal),
#      CT-FM-LP, SPECTRE (all endpoints), external validation, SPECTRE + Cox head (OS; check of the head)
#   3  SegRes-E2E predictions, then per-data reports (Table S4, subgroups, calibration, KM)
#   4  merged report -> outputs/<run>_study/report/ (+ spectre_cox_rows.md: the SPECTRE + Cox head rows of Tables S5-S7)
# Output: <outputs.dir>/<run>_study/  (.done or .failed appears there at the end)
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$HERE"
CONF="$HERE/pipeline/ctfm_config.py"
PY=${PY:-$(python3 "$CONF" get runtime.python 2>/dev/null || echo python3)}
PY=$(command -v "$PY") || { echo "python interpreter not found (set PY or runtime.python)"; exit 2; }
cfg() { "$PY" "$CONF" get "$1"; }
RUN=$(date +%Y%m%d_%H%M%S); GPUS=$(cfg runtime.gpus); SPECTRE_GPU=$(cfg runtime.spectre_gpu); DRY=0
OUTDIR=$(cfg outputs.dir); CDIR=$(cfg cache.dir)      # run folders / shared features and scan caches
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2;;
    --spectre-gpu) SPECTRE_GPU="$2"; shift 2;;
    --run) RUN="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown option: $1"; exit 2;;
  esac
done
ROOT="$OUTDIR/${RUN}_study"
SESSION="ctfm_${RUN}"
FIRST_GPU=$(echo $GPUS | awk '{print $1}')
NG=$(echo $GPUS | wc -w)

echo "run       $RUN"
echo "folder    $ROOT"
echo "session   $SESSION    SegRes-E2E GPUs: $GPUS    features GPU: $FIRST_GPU    SPECTRE GPU: $SPECTRE_GPU"
echo "python    $PY"
echo "config    $("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); from ctfm_config import CFG; print(CFG.file)' "$HERE/pipeline")"
echo "cache     $CDIR"
"$PY" "$CONF" check
for f in features_paper_st/features_maastro.npz features_paper_lp/features_maastro.npz features_paper_lp2/features_maastro.npz \
         features_spectre_paper/features_radcure.npz features_spectre_paper/features_maastro.npz \
         volumes/paper_lp/maastro.json volumes/paper_lp2/maastro.json; do
  [ -f "$CDIR/$f" ] && echo "   have   $f" || echo "   will make  $f"
done
{ nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"; } | sed 's/^/   GPU /'
if [ "$DRY" = 1 ]; then echo "DRY RUN -- nothing launched"; exit 0; fi
command -v tmux >/dev/null || { echo "tmux is required to launch the run"; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session $SESSION already exists"; exit 1; fi
mkdir -p "$ROOT/logs"
"$PY" "$CONF" dump "$ROOT/config_resolved.yaml" > /dev/null
# every job reads the configuration written above (and nothing from a stale tmux environment)
ENVSET="for v in \$(env | sed -n 's/^\(CTFM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset \"\$v\"; done; export CTFM_CONFIG=\"$ROOT/config_resolved.yaml\""
R="$PY $HERE/pipeline/runner.py"
SC="$PY $HERE/pipeline/run_spectre_cox.py"

cat > "$ROOT/segres_worker.sh" <<EOF
#!/usr/bin/env bash
# SegRes-E2E worker on GPU \$1: the OS/RFS networks (submitted data), then the LR/DM networks (corrected data).
# Both queues always run; the exit file records the worst exit code.
cd "$HERE"
$ENVSET
# One queue per (data, endpoint): each endpoint has its own input (runner.LP_INPUT).
g=\$1; worst=0
for q in "original os" "original efs" "corrected lr" "corrected dr"; do
  set -- \$q
  CUDA_VISIBLE_DEVICES=\$g $R segres-worker --data \$1 --endpoints \$2 --gpu \$g --root "$ROOT" 2>&1 | tee -a "$ROOT/logs/segres_gpu\$g.log"
  rc=\${PIPESTATUS[0]}; [ \$rc -gt \$worst ] && worst=\$rc
done
echo \$worst > "$ROOT/logs/.segres_gpu\$g.exit"
echo "SegRes-E2E worker on GPU \$g finished (exit \$worst)"
EOF

cat > "$ROOT/main.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "$HERE"
$ENVSET
log()  { echo "[\$(date '+%F %T')] \$*" | tee -a "$ROOT/logs/main.log"; }
stop() { log "FAILED: \$*"; touch "$ROOT/.failed"; exit 1; }
run()  { local name=\$1; shift; log "start \$name"; "\$@" > "$ROOT/logs/\$name.log" 2>&1 && log "done  \$name" || { log "FAILED \$name (logs/\$name.log)"; return 1; }; }

log "== step 0: inputs"
CACHE=""; SPEC=""
( for w in lp lp2; do [ -f "$CDIR/volumes/paper_\$w/maastro.json" ] || run segres_cache_\$w $R segres-cache --which \$w --procs 16 || exit 1; done ) &
CACHE=\$!
MISS=""; for c in Radcure Maastro; do [ -f "$CDIR/features_spectre_paper/features_\${c,,}.npz" ] || MISS="\$MISS \$c"; done
if [ -n "\$MISS" ]; then
  ( export CUDA_VISIBLE_DEVICES=$SPECTRE_GPU; run spectre_features $R run --data original --root "$ROOT" -- extract_spectre_features.py --preproc published --workers 8 --cohorts \$MISS || true
    [ -f "$CDIR/features_spectre_paper/features_radcure.npz" ] && [ -f "$CDIR/features_spectre_paper/features_maastro.npz" ] ) &
  SPEC=\$!
fi
for w in st lp lp2; do
  if [ ! -f "$CDIR/features_paper_\$w/features_maastro.npz" ]; then
    ( export CUDA_VISIBLE_DEVICES=$FIRST_GPU; run features_\$w $R features --which \$w ) || stop "features \$w"
  fi
done

log "== step 1: verification against the paper"
run verify $R verify --out "$ROOT/report" || log "!! verification did not pass -- continuing anyway (see report/verify.md)"

[ -z "\$CACHE" ] || wait \$CACHE || stop "SegRes scan cache"
[ -f "$CDIR/volumes/paper_lp/maastro.json" ] && [ -f "$CDIR/volumes/paper_lp2/maastro.json" ] || stop "SegRes scan cache missing"
log "== step 2: SegRes-E2E workers on GPUs $GPUS (windows gpuN) + CPU jobs"
for g in $GPUS; do tmux new-window -t "$SESSION" -n "gpu\$g" "bash '$ROOT/segres_worker.sh' \$g; exec bash"; done

declare -A LPIN=([os]=lp [dr]=lp [efs]=lp2 [lr]=lp2)     # CT-FM-LP input per endpoint (runner.LP_INPUT)
pids=()
for spec in "original os efs" "corrected lr dr"; do
  set -- \$spec; d=\$1; shift
  for ep in "\$@"; do for co in Radcure Maastro; do
    OMP_NUM_THREADS=8 run a1_\${d}_\${co}_\$ep $R run --data \$d --features st --root "$ROOT" -- run_a1.py --run $RUN --cohort \$co --endpoint \$ep --regime all &
    pids+=(\$!)
  done; done
  ( for ep in "\$@"; do run lp_\${d}_\$ep $R run --data \$d --features \${LPIN[\$ep]} --root "$ROOT" -- run_baselines.py --model ctfm_lp --run $RUN --endpoint \$ep || exit 1; done ) &
  pids+=(\$!)
  OMP_NUM_THREADS=16 run external_\$d $R run --data \$d --features st --root "$ROOT" -- run_external.py --run $RUN --spec primary &
  pids+=(\$!)
done
( while [ -n "\$SPEC" ] && kill -0 \$SPEC 2>/dev/null; do sleep 60; done     # SPECTRE features (sibling job)
  [ -f "$CDIR/features_spectre_paper/features_radcure.npz" ] && [ -f "$CDIR/features_spectre_paper/features_maastro.npz" ] || exit 1
  for job in "original os" "original efs" "corrected lr" "corrected dr"; do
    set -- \$job
    run spectre_\$1_\$2 $R run --data \$1 --root "$ROOT" -- run_baselines.py --model spectre --run $RUN --endpoint \$2 || exit 1
  done ) &
pids+=(\$!)
( while [ -n "\$SPEC" ] && kill -0 \$SPEC 2>/dev/null; do sleep 60; done     # SPECTRE + Cox head, OS only
  [ -f "$CDIR/features_spectre_paper/features_radcure.npz" ] && [ -f "$CDIR/features_spectre_paper/features_maastro.npz" ] || exit 1
  for co in Radcure Maastro; do OMP_NUM_THREADS=8 run spectre_cox_\$co $SC a1 --cohort \$co --root "$ROOT/spectre_cox" --run $RUN || exit 1; done
  OMP_NUM_THREADS=16 run spectre_cox_external $SC external --root "$ROOT/spectre_cox" --run $RUN || exit 1 ) &
pids+=(\$!)
fail=0; for p in "\${pids[@]}"; do wait \$p || fail=1; done
[ \$fail = 0 ] || stop "a CPU job failed -- see logs/"
log "CPU jobs finished; waiting for the SegRes-E2E workers"

while [ \$(ls "$ROOT"/logs/.segres_gpu*.exit 2>/dev/null | wc -l) -lt $NG ]; do sleep 120; done
grep -qv '^0$' "$ROOT"/logs/.segres_gpu*.exit && log "!! a SegRes worker exited with an error -- see logs/segres_gpu*.log"
log "== step 3: SegRes-E2E predictions and per-data reports"
for spec in "original os efs" "corrected lr dr"; do
  set -- \$spec; d=\$1; shift
  for ep in "\$@"; do run segres_\${d}_\$ep $R run --data \$d --root "$ROOT" -- run_baselines.py --model segres_e2e --run $RUN --endpoint \$ep || stop "segres \$d \$ep"; done
  A1=\$(ls -d "$ROOT"/results_\${d}_*$RUN | grep -v baselines | grep -v external | head -1)
  BASE=\$(ls -d "$ROOT"/results_baselines_\${d}_*$RUN | head -1)
  run table_s4_\$d  $R run --data \$d --root "$ROOT" -- make_table_s4.py --runs "\$A1" "\$BASE" --out "$ROOT/report_\$d" --no-figures || stop "table S4 \$d"
  run subgroups_\$d $R run --data \$d --root "$ROOT" -- run_subgroup.py --runs "\$A1" "\$BASE" --out "$ROOT/report_\$d" || log "!! subgroups \$d: figure or table failed (see logs)"
  [ -f "$ROOT/report_\$d/table_s5_subgroups_long.csv" ] || stop "subgroups \$d: no table"
  run calib_\$d     $R run --data \$d --root "$ROOT" -- make_calibration.py --runs "\$A1" "\$BASE" --out "$ROOT/report_\$d" || log "!! calibration \$d failed (non-fatal)"
  run km_\$d        $R run --data \$d --root "$ROOT" -- make_km.py --runs "\$A1" --out "$ROOT/report_\$d/figures" || log "!! KM \$d failed (non-fatal)"
done
log "== step 4: merged report"
run report $R run --data corrected --features st --root "$ROOT" -- merged_report.py --root "$ROOT" || stop "merged report"
run spectre_cox_rows $SC rows --root "$ROOT/spectre_cox" --main "$ROOT" --out "$ROOT/report" || log "!! SPECTRE + Cox head rows failed (see logs)"
touch "$ROOT/.done"
log "ALL DONE -> $ROOT/report"
EOF
chmod +x "$ROOT/main.sh" "$ROOT/segres_worker.sh"
tmux new-session -d -s "$SESSION" -n main "bash '$ROOT/main.sh'; exec bash"
echo "launched: tmux attach -t $SESSION     (log: $ROOT/logs/main.log)"
