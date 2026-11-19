# CT-FM oropharyngeal-cancer prognosis pipeline: the pinned environment of the published results.
# See README, "Running with Docker". The PyTorch wheel brings its own CUDA 11.8 runtime: the host needs only an
# NVIDIA driver and the NVIDIA Container Toolkit (docker run --gpus ...).
#
#   docker build -t ctfm-opc-prognosis .                              # with the SPECTRE packages (default)
#   docker build --build-arg WITH_SPECTRE=0 -t ctfm-opc-prognosis .   # without them
FROM python:3.10.17-slim-bookworm

ARG WITH_SPECTRE=1
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/hf_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install -y --no-install-recommends tmux procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt constraints.txt ./
# PyTorch 2.5.1 and torchvision 0.20.1 built for CUDA 11.8 (and setuptools as in the original environment)
# first, then the pinned packages; the optional SPECTRE packages are the commented lines at the end of
# requirements.txt. constraints.txt pins every sub-dependency (torch and torchvision included) to the version
# of the original environment. spectre-fm goes in without its
# dependencies: it asks for monai[all], whose extras clash with those versions and which SPECTRE does not use.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 setuptools==75.8.0 -c constraints.txt \
        --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple \
 && pip install -r requirements.txt -c constraints.txt \
 && if [ "$WITH_SPECTRE" = "1" ]; then \
      sed -n '/^# --- OPTIONAL: SPECTRE/,$ s/^# \([A-Za-z0-9_.-]*==[^ ]*\).*/\1/p' requirements.txt \
          > /tmp/spectre-requirements.txt \
      && pip install -r /tmp/spectre-requirements.txt -c constraints.txt \
      && pip install --no-deps spectre-fm==0.1.0; \
    fi \
 && python -c "import torch, torchvision; v = (torch.__version__, torchvision.__version__); \
assert v == ('2.5.1+cu118', '0.20.1+cu118'), v"

COPY . .
# mount points (see README); writable for the --user the container runs as
RUN mkdir -p /hf_cache /app/data /app/cache /app/outputs \
 && chmod 777 /hf_cache /app/data /app/cache /app/outputs

# with --user the account has no home folder: give it a writable one (caches of libraries)
ENV HOME=/tmp
CMD ["bash"]
