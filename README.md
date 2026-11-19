# CTFM-OPC-Prognosis: CT foundation-model representations for survival prediction in head and neck (oropharyngeal) cancer

Code for the analyses of *Do General-Purpose CT Foundation Models Add Prognostic Value Beyond Clinical Factors in Oropharyngeal
Cancer? A Multi-Endpoint Evaluation with External Validation* (citation below: TBD). It trains and evaluates survival models for four endpoints on two
public cohorts (RADCURE and Maastro, from The Cancer Imaging Archive) and produces the paper's figures and
tables, every estimate with a 95% bootstrap confidence interval.

| Model | What it is |
|---|---|
| **CT-FM-ST** | frozen CT-FM encoder (`project-lighter/ct_fm_feature_extractor`), 512 pooled features, ridge Cox model (lifelines) |
| **CT-FM-LP** | frozen CT-FM encoder features from this model's own extraction, one linear Cox layer trained with PyTorch |
| **SegRes-E2E** | the CT-FM architecture (SegResEncoder) trained from scratch, linear Cox head |
| **SPECTRE** | frozen SPECTRE ViT-Large features (`spectre-fm`), linear Cox head |
| **SPECTRE + Cox head** | the same SPECTRE features with the CT-FM-ST ridge Cox head (OS only; a check of the prediction head) |
| **Clinical only** | Cox model: age, sex, HPV, smoking, T stage, N stage (AJCC) |
| **Clinical + CT-FM-ST** | the clinical Cox model plus the out-of-fold CT-FM-ST risk score |

Endpoints: overall survival (OS), recurrence-free survival (RFS), locoregional failure (LR/LRF) and distant
metastasis (DM). In the code they are `os`, `efs`, `lr` and `dr`. OS and RFS use the cohort tables of the
submitted manuscript ("original"); LR and DM use the corrected tables with pure LR / DM events, death censored
("corrected", made by `data_preparation/`).

Protocol: one stratified 70/15/15 train/validation/test split per cohort and endpoint (seed 42); training on
subsets of 10, 24, 50, 100, ... patients ("few shot", 5 random subsets per size) and on the whole training
split ("full shot"); evaluation on the held-out test set (Radcure n = 202, Maastro n = 26) with C-index and integrated Brier score
(IBS); 95% CIs from 2,000 event-stratified bootstrap resamples of the test patients; subgroups (sex, HPV,
T stage, N stage); calibration at 2 years; external validation trained on Radcure and tested on all 173 Maastro
patients; paired bootstrap comparisons between models.

## What it produces

The final folder (`outputs.final_dir`, written by `pipeline/make_paper_style_figures.py`). File names follow the
numbering of the submitted manuscript.

| File | Content |
|---|---|
| `figures/Figure2_fullshot_cindex_95CI` | full-shot C-index, internal test sets, six models (SPECTRE: OS only) |
| `figures/Figure3_fewshot_curves_95CI` | few-shot learning curves with 95% CI bands |
| `figures/Figure4_subgroups_cindex` | subgroup C-index, full shot |
| `figures/Figure5a_calibration_Radcure`, `Figure5b_calibration_Maastro` | 2-year calibration and IBS, five models |
| `figures/FigureExt_external_*` | external validation: C-index bars, C-index / IBS forest plot, Kaplan-Meier by risk tertile |
| `figures/FigureKM_internal_tertiles_{Radcure,Maastro}` | Kaplan-Meier curves by predicted-risk tertile, internal test sets |
| `figures/FigureKM_cohort_all_endpoints` | whole-cohort Kaplan-Meier curves with numbers at risk |
| `tables/TableS4_internal_cindex_95CI` | C-index at 10 / 24 / 50 / 100 / full training patients |
| `tables/TableS5_subgroups_cindex_95CI` | subgroup C-index |
| `tables/TableS6_internal_ibs_calibration_95CI` | C-index, IBS, calibration error (full shot) |
| `tables/TableExt_external_validation_95CI` | external validation: C-index, IBS, ΔC-index vs clinical model |
| `tables/TablePaired_model_comparisons_95CI` | paired C-index differences between models (internal and external) |
| `tables/TableSens_RFS_corrected_dates` | RFS with the submitted vs the corrected Radcure event dates |

Also: contribution maps of the CT-FM-ST OS model and a confounding analysis
(`<outputs.dir>/explainability_maps/`), and per-data-version reports with every intermediate table
(`<run>/report_original/`, `<run>/report_corrected/`, `<run>/report/`).

## Repository layout

```
config.yaml               configuration: input locations, cache, outputs, GPUs (the only file to edit)
run_pipeline.sh           the main run: features, all six models, few/full shot, subgroups, external, reports
run_final_figures.sh      post-processing of a finished run: calibration, sensitivity, SegRes refit, final figures, explainability
data_preparation/         builds the corrected cohort tables (pure LR / DM events, RFS times)
pipeline/                 the analysis code (entry point: runner.py; configuration: ctfm_config.py)
optional/                 score_paper_networks_ibs.py -- needs the submitted study's saved networks
tests/                    synthetic data generator and CPU smoke test
```

Main scripts in `pipeline/`:

| Script | Step |
|---|---|
| `runner.py` | sets the inputs of each model and runs the target scripts (`features`, `segres-cache`, `segres-worker`, `run`, `verify`) |
| `extract_features.py`, `extract_spectre_features.py` | CT-FM / SPECTRE feature extraction (GPU) |
| `train_segres.py`, `segres_cache.py` | SegRes-E2E training (GPU workers) and their scan cache |
| `run_a1.py` | CT-FM-ST, Clinical only, Clinical + CT-FM-ST (few and full shot) |
| `run_baselines.py` | CT-FM-LP, SPECTRE, SegRes-E2E predictions |
| `run_external.py` | external validation Radcure → Maastro |
| `run_spectre_cox.py` | SPECTRE + Cox head (OS): internal and external runs, and its rows of Tables S5–S7 |
| `make_table_s4.py`, `run_subgroup.py`, `make_calibration.py` / `make_calibration_safe.py`, `make_km.py` | per-data-version tables and figures |
| `merged_report.py` | merged report (OS/RFS from the original data, LR/DM from the corrected data) |
| `refit_segres_lrf_dm.py` | SegRes-E2E LR/DM survival probabilities for calibration / IBS (GPU, inference only) |
| `make_paper_style_figures.py` | the final figures and tables |
| `explainability_maps.py` | contribution maps and confounding check (GPU) |
| `cohort.py`, `models.py`, `bootstrap_ci.py`, `ipcw_brier.py`, `subsample.py`, ... | shared code |

## Requirements

* Linux, Python 3.10, and the pinned packages of `requirements.txt` (or `environment.yml`):

  ```
  pip install -r requirements.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu118
  # or
  mamba env create -f environment.yml && conda activate ctfm-opc
  ```
  SPECTRE needs the optional packages listed at the end of `requirements.txt`, and then
  `pip install --no-deps spectre-fm==0.1.0`.
* NVIDIA GPU(s) for feature extraction, SegRes-E2E training (300 networks, one worker per GPU), the SegRes-E2E
  refit and the explainability step. Everything else runs on the CPU. The main run of the published results
  took about 22 hours with three GPUs.
* `tmux` (the main launcher runs in a tmux session).
* Or use the Docker image, which contains all of the above (see *Running with Docker*).
* Model weights, downloaded on first use: CT-FM from HuggingFace (`project-lighter/ct_fm_feature_extractor`,
  configurable as `model.ctfm_weights`), SPECTRE through the `spectre-fm` package.

## Input data

The two cohorts are available from The Cancer Imaging Archive (https://www.cancerimagingarchive.net):
RADCURE (Welch et al., Med Phys 2024, https://doi.org/10.1002/mp.16972) and the Maastro head-and-neck cohort.
No data are included in this repository.

## Configuration

Everything that depends on the machine is in `config.yaml`; relative paths are taken from the folder of the
configuration file. Any value can be overridden with an environment variable `CTFM_<SECTION>_<KEY>`, and
another file can be used with `CTFM_CONFIG`:

```
CTFM_DATA_RADCURE_IMAGES=/data/radcure/images CTFM_RUNTIME_GPUS="0 1 2" bash run_pipeline.sh --dry-run
CTFM_CONFIG=/path/to/my_config.yaml bash run_pipeline.sh --dry-run

python pipeline/ctfm_config.py show     # the resolved configuration
python pipeline/ctfm_config.py check    # which configured inputs exist
```

| Section | Keys |
|---|---|
| `data` | the cohort tables, scan and mask folders |
| `cache` | `dir`: CT-FM / SPECTRE features and pre-processed scan arrays, shared by runs |
| `outputs` | `dir`: run folders; `final_dir`: final figures and tables; `paper_run`: run to post-process (null = newest) |
| `model` | `ctfm_weights`: HuggingFace id or local folder of the CT-FM encoder |
| `submitted` | optional |
| `runtime` | `python`, `gpus` (SegRes-E2E workers), `spectre_gpu`, `device` (CT-FM feature extraction) |
| `testing` | smoke test only; keep the defaults for real runs |

The analysis settings (split, seeds, training sizes, models, bootstrap) are fixed in the code; they are the
ones that produced the published results.

## Running

```
# 0. corrected cohort tables (once)
python data_preparation/build_radcure_corrected.py
python data_preparation/build_maastro_corrected.py

# 1. main run: check the plan and the inputs, then launch (tmux session ctfm_<run id>)
bash run_pipeline.sh --gpus "0 1 2" --spectre-gpu 2 --dry-run
bash run_pipeline.sh --gpus "0 1 2" --spectre-gpu 2
#    resume a stopped run (finished steps are kept):  bash run_pipeline.sh --run <run id> --gpus "0 1 2"

# 2. post-processing of the finished run (foreground; --steps runs a subset)
bash run_final_figures.sh --dry-run
bash run_final_figures.sh --gpu 0
```

`run_pipeline.sh` steps (logs in `<outputs.dir>/<run>_study/logs/`, `.done` or `.failed` at the end):

0. CT-FM features for the three model inputs, SPECTRE features, SegRes-E2E scan caches (GPU / CPU)
1. `runner.py verify` (optional; needs `submitted.workspace`, skipped otherwise)
2. SegRes-E2E networks on the GPUs; in parallel on the CPU: CT-FM-ST, clinical and combined models (`run_a1.py`),
   CT-FM-LP and SPECTRE (`run_baselines.py`), external validation (`run_external.py`)
3. SegRes-E2E predictions, then Table S4, subgroups, calibration and Kaplan-Meier per data version
4. merged report (`merged_report.py`) → `<run>/report/`

`run_final_figures.sh` steps: `calibration` (`make_calibration_safe.py`, both data versions), `sensitivity`
(RFS with the corrected Radcure dates), `saved-networks` (optional), `refit` (`refit_segres_lrf_dm.py`, GPU),
`figures` (`make_paper_style_figures.py` → `outputs.final_dir`), `explainability` (`explainability_maps.py`, GPU).

Single steps can be run through `runner.py`, which sets each model's input before running a script:

```
python pipeline/runner.py features --which st          # CT-FM features, CT-FM-ST input
python pipeline/runner.py run --data original --features st --root <run folder> -- \
       run_a1.py --run <run id> --cohort Radcure --endpoint os --regime all
python pipeline/runner.py run --data original --root <run folder> -- \
       make_table_s4.py --runs <run folder>/results_original_n8_<run id> --out <run folder>/report_original
```

## Running with Docker

The image contains Python 3.10, the pinned packages (including the optional SPECTRE packages) and `tmux`; the
PyTorch wheel brings its own CUDA 11.8 runtime, so the host needs only an NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/). Data, caches and
outputs stay on the host and are mounted into the container, at the locations the default `config.yaml` expects.

```
# 1. build the image (once; about 8 GB)
docker build -t ctfm-opc-prognosis .
#    without the SPECTRE packages:  docker build --build-arg WITH_SPECTRE=0 -t ctfm-opc-prognosis .

# 2. check it: synthetic CPU smoke test, no data needed (about 10 minutes; downloads the CT-FM weights)
docker run --rm ctfm-opc-prognosis bash tests/smoke_test.sh --work /tmp/smoke

# 3. start a container that keeps running, with your folders mounted
docker run -d --name ctfm --gpus all --shm-size=16g --user "$(id -u):$(id -g)" \
    -v /path/to/data:/app/data \
    -v /path/to/cache:/app/cache \
    -v /path/to/outputs:/app/outputs \
    -v /path/to/hf_cache:/hf_cache \
    ctfm-opc-prognosis sleep infinity

# 4. run the pipeline inside it (same commands as in "Running")
docker exec ctfm bash run_pipeline.sh --gpus "0 1 2" --spectre-gpu 2 --dry-run
docker exec ctfm bash run_pipeline.sh --gpus "0 1 2" --spectre-gpu 2
docker exec -it ctfm tmux attach -t ctfm_<run id>     # follow the run; detach with Ctrl-b d
docker exec ctfm bash run_final_figures.sh            # after the main run has finished

# 5. stop the container when everything is done (the outputs stay in /path/to/outputs)
docker rm -f ctfm
```

* `/path/to/data` holds the cohort tables and image folders under the names in `config.yaml` (e.g.
  `new_radcure_data.csv`, `imagesTs_radcure/`); another layout can be set with `-e CTFM_DATA_RADCURE_IMAGES=...`
  or a mounted configuration file (`-e CTFM_CONFIG=/app/data/my_config.yaml`).
* `/path/to/hf_cache` keeps the downloaded CT-FM and SPECTRE weights between containers.
* `--gpus all` makes the host's GPUs visible with their host numbers; use e.g. `--gpus '"device=0,2"'` to
  expose only some of them (they are then numbered 0, 1, ... inside the container).
* The container runs the pipeline as your user (`--user`), so the outputs on the host belong to you.
* The image has the package versions of the environment that produced the published results
  (`constraints.txt`); the smoke test and the SPECTRE features give byte-identical results in both.

## Outputs

```
<outputs.dir>/<run id>_study/
    results_{original,corrected}_n8_<run id>/            CT-FM-ST, Clinical only, Clinical + CT-FM-ST: per-cell results, predictions, models
    results_baselines_{original,corrected}_n8_<run id>/  CT-FM-LP, SPECTRE, SegRes-E2E predictions
    results_external_{original,corrected}_n8_<run id>/   external validation
    segres_e2e/                                          trained SegRes-E2E networks
    report_{original,corrected}/                         Table S4 / S5 / S6 inputs per data version
    report/                                              merged report; report/paper_style/ = final figures and tables
    sensitivity_rfs_corrected_dates/                     RFS sensitivity run
    spectre_cox/                                         SPECTRE + Cox head (OS); its table rows: report/spectre_cox_rows.md
    logs/, config_resolved.yaml
<outputs.dir>/segres_refit_lrf_dm/, paper_networks_ibs/ (optional), explainability_maps/
<outputs.final_dir>/figures/, tables/
<cache.dir>/features_paper_{st,lp,lp2}/, features_spectre_paper/, volumes/
```

Every result folder holds `run_info.json` (data files with md5, package versions, the resolved configuration)
and `code_snapshot/` (the scripts that produced it).

## Citation

TBD

## License

CC BY-NC-ND 4.0
