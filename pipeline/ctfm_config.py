"""One configuration for the whole pipeline: config.yaml plus environment-variable overrides.

Where the values come from, in order (later wins):
  1. <repository>/config.yaml, or the file named by the CTFM_CONFIG environment variable;
  2. environment variables CTFM_<SECTION>_<KEY>, e.g.
         CTFM_DATA_RADCURE_IMAGES=/data/radcure/images   (data.radcure_images)
         CTFM_RUNTIME_GPUS="0 1"                          (runtime.gpus)
     The value is parsed as YAML (so "null", "true", "20" work).
Relative paths are taken from the folder of the configuration file.

    python ctfm_config.py show                  # the resolved configuration
    python ctfm_config.py get runtime.gpus      # one value (paths are printed absolute)
    python ctfm_config.py check                 # which configured inputs exist
    python ctfm_config.py run                   # the run folder to post-process
    python ctfm_config.py dump <file>           # write the resolved configuration (absolute paths)

Settings that change results are NOT configurable here: the pipeline's numerical settings
live in the code, as they produced the published numbers. The `testing` section exists only
for the synthetic smoke test (tests/smoke_test.sh) and must stay at its defaults otherwise.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PIPELINE = REPO / "pipeline"

# Keys whose values are filesystem paths (resolved relative to the configuration file).
PATH_KEYS = {
    "data.radcure_csv", "data.maastro_csv", "data.radcure_csv_corrected", "data.maastro_csv_corrected",
    "data.maastro_pure_events_csv", "data.radcure_images", "data.maastro_images",
    "data.radcure_masks", "data.maastro_masks",
    "cache.dir", "outputs.dir", "outputs.final_dir",
    "submitted.workspace", "submitted.paper_md",
}

DEFAULTS = {
    "data": {"radcure_csv": None, "maastro_csv": None, "radcure_csv_corrected": None,
             "maastro_csv_corrected": None, "maastro_pure_events_csv": None,
             "radcure_images": None, "maastro_images": None, "radcure_masks": None, "maastro_masks": None},
    "cache": {"dir": "cache"},
    "outputs": {"dir": "outputs", "final_dir": "outputs/figures_and_tables", "paper_run": None},
    "model": {"ctfm_weights": "project-lighter/ct_fm_feature_extractor"},
    "submitted": {"workspace": None, "paper_md": None, "use_saved_segres_networks": "auto"},
    "runtime": {"python": "python3", "gpus": "0", "spectre_gpu": 0, "device": "auto"},
    "testing": {"random_init_encoder": False, "bootstrap_b": None},
}


class Config:
    def __init__(self, file: Path | None = None):
        self.file = Path(file or os.environ.get("CTFM_CONFIG") or REPO / "config.yaml").resolve()
        raw = {}
        if self.file.exists():
            raw = yaml.safe_load(self.file.read_text()) or {}
        elif "CTFM_CONFIG" in os.environ:
            raise FileNotFoundError(f"CTFM_CONFIG={self.file} does not exist")
        self.values = {s: dict(v) for s, v in DEFAULTS.items()}
        for sec, kv in raw.items():
            if not isinstance(kv, dict):
                raise ValueError(f"{self.file}: section '{sec}' must be a mapping")
            self.values.setdefault(sec, {}).update(kv)
        for sec, kv in self.values.items():               # environment overrides
            for key in kv:
                env = f"CTFM_{sec}_{key}".upper()
                if env in os.environ:
                    kv[key] = yaml.safe_load(os.environ[env]) if os.environ[env] != "" else None

    def get(self, dotted: str, default=None):
        sec, key = dotted.split(".", 1)
        v = self.values.get(sec, {}).get(key, default)
        return default if v is None else v

    def path(self, dotted: str, required: bool = False) -> Path | None:
        """A configured path, absolute (relative values are taken from the config file's folder)."""
        v = self.get(dotted)
        if v in (None, ""):
            if required:
                raise SystemExit(f"configuration: '{dotted}' is not set (config file {self.file}, "
                                 f"or environment variable CTFM_{dotted.replace('.', '_').upper()})")
            return None
        p = Path(os.path.expandvars(os.path.expanduser(str(v))))
        return p if p.is_absolute() else (self.file.parent / p).resolve()

    def resolved(self) -> dict:
        out = {}
        for sec, kv in self.values.items():
            out[sec] = {}
            for key, v in kv.items():
                d = f"{sec}.{key}"
                out[sec][key] = str(self.path(d)) if (d in PATH_KEYS and v not in (None, "")) else v
        return out


CFG = Config()


# --------------------------------------------------------------------------- helpers

def data_path(dotted: str) -> Path:
    """A data path that may be unset at import time (scripts fail with a clear message when used)."""
    p = CFG.path(dotted)
    return p if p is not None else Path(f"<{dotted} not configured>")


def cache_dir() -> Path:
    return CFG.path("cache.dir")


def outputs_dir() -> Path:
    return CFG.path("outputs.dir")


def paper_run(must_exist: bool = True):
    """(run folder, run id) of the run to post-process.

    outputs.paper_run = <YYYYMMDD_HHMMSS> picks one; unset = the newest <outputs>/<id>_study.
    With must_exist=False a missing run gives a placeholder (for module-level defaults)."""
    rid = CFG.get("outputs.paper_run")
    if rid is not None:
        rid = str(rid)          # YAML reads an unquoted 20261104_054739 as the integer 20261104054739
        if rid.isdigit() and len(rid) == 14:
            rid = f"{rid[:8]}_{rid[8:]}"
    if rid:
        root = outputs_dir() / f"{rid}_study"
    else:
        hits = sorted(p for p in outputs_dir().glob("*_study") if p.is_dir()) if outputs_dir().exists() else []
        root = hits[-1] if hits else outputs_dir() / "NO_RUN_YET_study"
    if must_exist and not root.is_dir():
        raise SystemExit(f"no run folder {root} -- run run_pipeline.sh first, "
                         f"or set outputs.paper_run")
    return root, root.name.replace("_study", "")


def torch_device():
    """runtime.device: auto (CUDA when available, else CPU), cuda or cpu."""
    import torch
    d = str(CFG.get("runtime.device", "auto")).lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def ctfm_encoder():
    """The frozen CT-FM encoder (HuggingFace model.ctfm_weights).

    testing.random_init_encoder = true gives a RANDOMLY INITIALISED encoder of the same
    architecture instead -- for the synthetic smoke test only, when the weights cannot be
    downloaded. Its features are meaningless."""
    from lighter_zoo import SegResEncoder
    if CFG.get("testing.random_init_encoder", False):
        import torch
        print("!! TEST MODE (testing.random_init_encoder): randomly initialised CT-FM encoder -- "
              "features are meaningless", file=sys.stderr, flush=True)
        torch.manual_seed(0)
        return SegResEncoder(spatial_dims=3, in_channels=1, init_filters=32, blocks_down=(1, 2, 2, 4, 4))
    return SegResEncoder.from_pretrained(CFG.get("model.ctfm_weights"))


def test_bootstrap_b():
    """testing.bootstrap_b: number of bootstrap resamples for smoke tests (None = the code's own)."""
    b = CFG.get("testing.bootstrap_b")
    return int(b) if b else None


def check() -> list[tuple[str, str, bool]]:
    rows = []
    for d in sorted(PATH_KEYS):
        p = CFG.path(d)
        rows.append((d, str(p) if p else "(not set)", bool(p and p.exists())))
    return rows


def _main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "show":
        print(f"# configuration file: {CFG.file}")
        print(yaml.safe_dump(CFG.resolved(), sort_keys=False).rstrip())
    elif cmd == "get":
        d = argv[1]
        v = CFG.path(d) if d in PATH_KEYS else CFG.get(d)
        print("" if v is None else v)
    elif cmd == "check":
        for d, p, ok in check():
            print(f"   {'ok     ' if ok else 'missing'}  {d:<32} {p}")
    elif cmd == "run":
        print(paper_run()[0])
    elif cmd == "dump":
        out = Path(argv[1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"# resolved from {CFG.file} (+ CTFM_* environment overrides)\n"
                       + yaml.safe_dump(CFG.resolved(), sort_keys=False))
        print(out)
    else:
        print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
