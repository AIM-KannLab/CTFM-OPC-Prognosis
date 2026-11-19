#!/usr/bin/env bash
# Post-processing of a finished run (run_pipeline.sh): the final tables and figures, in the order
# they were made for the paper. Runs in the foreground (wrap it in tmux/screen for long steps).
#
#   bash run_final_figures.sh --dry-run                       # show the plan and check inputs; run nothing
#   bash run_final_figures.sh                                 # every step, newest run (or outputs.paper_run)
#   bash run_final_figures.sh --run 20261104_054739 --gpu 0   # a given run; GPU of the GPU steps
#   bash run_final_figures.sh --steps "calibration figures"   # only some steps
#
# Steps (log: <run>/logs/final_<step>[_<data>].log):
#   calibration     Table S6 / Figure 5 inputs per data version (original: OS/RFS, corrected: LR/DM) with
#                   make_calibration_safe.py -> <run>/report_{original,corrected}/                        (CPU)
#   sensitivity     RFS with the corrected Radcure event dates (run_a1.py, full shot, corrected data)
#                   -> <run>/sensitivity_rfs_corrected_dates/                                             (CPU)
#   saved-networks  OPTIONAL: SegRes-E2E OS/RFS IBS and calibration of the submitted study's own saved networks
#                   (optional/score_paper_networks_ibs.py) -> <outputs.dir>/paper_networks_ibs/          (GPU)
#                   Runs only when submitted.workspace is set and use_saved_segres_networks is not off.
#   refit           SegRes-E2E LR/DM survival probabilities with the submitted refit step
#                   (refit_segres_lrf_dm.py) -> <outputs.dir>/segres_refit_lrf_dm/                       (GPU)
#   figures         make_paper_style_figures.py -> <run>/report/paper_style/ and outputs.final_dir        (CPU)
#   explainability  explainability_maps.py (contribution maps, confounding check)
#                   -> <outputs.dir>/explainability_maps/                                           (GPU)
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$HERE"
CONF="$HERE/pipeline/ctfm_config.py"
PY=${PY:-$(python3 "$CONF" get runtime.python 2>/dev/null || echo python3)}
PY=$(command -v "$PY") || { echo "python interpreter not found (set PY or runtime.python)"; exit 2; }
cfg() { "$PY" "$CONF" get "$1"; }
ALL="calibration sensitivity saved-networks refit figures explainability"
STEPS="$ALL"; DRY=0; RUN=""; GPU=$(cfg runtime.gpus | awk '{print $1}')
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown option: $1"; exit 2;;
  esac
done
for s in $STEPS; do case " $ALL " in *" $s "*) ;; *) echo "unknown step: $s (steps: $ALL)"; exit 2;; esac; done
[ -n "$RUN" ] && export CTFM_OUTPUTS_PAPER_RUN="\"$RUN\""      # quoted: YAML would read the id as a number
ROOT=$("$PY" "$CONF" run)                                     # stops if there is no such run folder
RUN=$(basename "$ROOT" _study)
OUTDIR=$(cfg outputs.dir)
R="$PY $HERE/pipeline/runner.py"
want() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
a1_dir()   { ls -d "$ROOT"/results_"$1"_*"$RUN" 2>/dev/null | grep -v baselines | grep -v external | head -1 || true; }
base_dir() { ls -d "$ROOT"/results_baselines_"$1"_*"$RUN" 2>/dev/null | head -1 || true; }

WS=$(cfg submitted.workspace); USE=$(cfg submitted.use_saved_segres_networks)
SAVED=0
if [ -n "$WS" ] && [ -f "$WS/generate_calibration_pytorch_head.py" ] && [ "$USE" != "off" ] && [ "$USE" != "False" ]; then SAVED=1; fi

echo "run       $RUN"
echo "folder    $ROOT"
echo "outputs   $OUTDIR      final figures/tables: $(cfg outputs.final_dir)"
echo "python    $PY      GPU steps on GPU $GPU"
echo "steps     $STEPS"
for d in original corrected; do
  a=$(a1_dir $d); b=$(base_dir $d)
  echo "   $d: ${a:-MISSING results_${d}_*$RUN}  |  ${b:-MISSING results_baselines_${d}_*$RUN}"
done
[ -f "$ROOT/report/table_s4_cindex_long.csv" ] && echo "   have   report/ (merged report)" || echo "   MISSING report/ -- did run_pipeline.sh finish?"
if [ "$SAVED" = 1 ]; then echo "   saved-networks: will run (submitted.workspace = $WS)"
else echo "   saved-networks: skipped (submitted.workspace not set / no submitted code there, or use_saved_segres_networks = off)"; fi
if [ "$DRY" = 1 ]; then echo "DRY RUN -- nothing run"; exit 0; fi

mkdir -p "$ROOT/logs"
step() { local name=$1; shift; echo "[$(date '+%F %T')] start $name"; "$@" > "$ROOT/logs/final_$name.log" 2>&1 \
         && echo "[$(date '+%F %T')] done  $name" || { echo "[$(date '+%F %T')] FAILED $name (logs/final_$name.log)"; return 1; }; }

if want calibration; then
  for d in original corrected; do
    [ -n "$(a1_dir $d)" ] && [ -n "$(base_dir $d)" ] || { echo "calibration $d: run folders missing (see the dry run)"; exit 1; }
    # non-fatal: the tables are written before the figure; checked right after
    step calibration_$d $R run --data $d --root "$ROOT" -- make_calibration_safe.py \
         --runs "$(a1_dir $d)" "$(base_dir $d)" --out "$ROOT/report_$d" || true
    for f in table_s6_calibration_long.csv fig5_calibration_bins.csv; do
      [ -f "$ROOT/report_$d/$f" ] || { echo "calibration $d: $f was not written -- see logs/final_calibration_$d.log"; exit 1; }
    done
  done
fi
if want sensitivity; then
  S="$ROOT/sensitivity_rfs_corrected_dates"; mkdir -p "$S/logs"
  for co in Radcure Maastro; do
    echo "[$(date '+%F %T')] start sensitivity $co"
    OMP_NUM_THREADS=8 $R run --data corrected --features st --root "$S" -- run_a1.py --run "$RUN" --cohort $co \
        --endpoint efs --regime full > "$S/logs/a1_corrected_${co}_efs.log" 2>&1 \
      || { echo "sensitivity $co failed -- see $S/logs/a1_corrected_${co}_efs.log"; exit 1; }
  done
  echo "[$(date '+%F %T')] done  sensitivity"
fi
if want saved-networks; then
  if [ "$SAVED" = 1 ]; then
    CUDA_VISIBLE_DEVICES=$GPU step saved_networks "$PY" "$HERE/optional/score_paper_networks_ibs.py" || exit 1
  else
    echo "saved-networks skipped (see above): Table S6 / Figure 5 use the re-trained SegRes-E2E OS/RFS networks"
  fi
fi
if want refit; then
  CUDA_VISIBLE_DEVICES=$GPU step refit "$PY" "$HERE/pipeline/refit_segres_lrf_dm.py" || exit 1
fi
if want figures; then
  step figures "$PY" "$HERE/pipeline/make_paper_style_figures.py" --report "$ROOT/report" || exit 1
fi
if want explainability; then
  CUDA_VISIBLE_DEVICES=$GPU step explainability "$PY" "$HERE/pipeline/explainability_maps.py" || exit 1
fi
echo "done -> $ROOT/report/paper_style  and  $(cfg outputs.final_dir)"
