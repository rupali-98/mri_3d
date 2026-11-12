# Docker: one container per part of the project

Research use only, not for diagnosis. Back to the [README](../README.md).

The project runs natively in a venv (`docs/usage.md`). Docker describes the same environment as code, so the
viewer, the API and the MLflow UI start the same way on another machine, and training runs on a Linux/NVIDIA
GPU machine without installing CUDA by hand. On macOS, train in the native venv: Docker can't reach the
Apple GPU (MPS).

## Layout

```
mri_3d/
├── frontend/
│   ├── Dockerfile                 Streamlit viewer (streamlit_dashboard/), CPU torch
│   └── Dockerfile.dockerignore
├── backend/
│   ├── Dockerfile                 FastAPI service (src/hippo/api.py) + model package, CPU; the Cloud Run image
│   └── Dockerfile.dockerignore
├── model-service/
│   ├── Dockerfile                 training and evaluation (hippo.train, evaluate, ...), CUDA torch, GPU
│   └── Dockerfile.dockerignore
├── compose.yaml                   the services, ports, volumes, health checks, GPU, secret
├── .env.example                   ports and the torch build; copy to .env (git-ignored)
└── secrets/api_key.txt            the API key (git-ignored, you create it)
```

The Python code stays in `src/hippo/` and `streamlit_dashboard/`: the tests, CI and `PYTHONPATH=src` use it
there. Each folder holds its service's Dockerfile, and each Dockerfile copies only the code it needs.

| Service | What it is in this project | Image | GPU |
|---|---|---|---|
| `frontend` | The viewer page. It predicts in-process with the runs in `runs/`, because it compares several runs with the ground truth and the API serves one packaged model | `frontend/Dockerfile` | no |
| `backend` | The HTTP API: input checks, API key, QC flags, and the model inference itself (well under a second on CPU) | `backend/Dockerfile` | no |
| `model-service` | The heavy model work: training, cross-validation, evaluation, external tests | `model-service/Dockerfile` | yes (NVIDIA) |
| `mlflow` | Experiment tracking UI over `runs/mlflow.db`; like a database, an off-the-shelf image, no Dockerfile | `ghcr.io/mlflow/mlflow:v3.17.1` | no |

## Run it

```bash
mkdir -p secrets && openssl rand -hex 32 > secrets/api_key.txt    # once; the backend needs it
cp .env.example .env                                                # optional: change ports
python -m hippo.package_model --experiment runs/baseline --src data/mri_dataset --single   # backend's model
docker compose up --build              # frontend :8501, backend :8080, mlflow :5001
docker compose up --watch              # same; code edits sync into the running containers
docker compose ps                      # health of each service
docker compose logs -f backend         # one log line per request
docker compose down
```

Calling the API: `curl -H "X-API-Key: $(cat secrets/api_key.txt)" -F file=@scan.nii.gz http://localhost:8080/segment/crop`.

Training on a GPU machine (needs the NVIDIA driver and the NVIDIA Container Toolkit on the host; check with
`nvidia-smi` first):

```bash
docker compose --profile gpu run --rm model-service python -m hippo.train --config configs/baseline.yaml
docker compose --profile gpu run --rm model-service python -m hippo.evaluate --run runs/baseline
```

## Choices

- **Data stays on the host.** `data/` and `runs/` are bind mounts; no image contains scans, labels or runs.
  Each Dockerfile has its own `Dockerfile.dockerignore` that allows only the files it copies, so nothing
  else (`.env`, `secrets/`, `.venv/`) reaches the build.
- **The model package is small** (a few MB), so the backend image bakes it in for Cloud Run. Compose mounts
  `deploy/model/` over it, so a repackaged model is used after `docker compose restart backend`.
- **Secrets as files.** The API key is a Compose secret (`/run/secrets/api_key`, read via
  `HIPPO_API_KEY_FILE`), not an environment variable that `docker inspect` would show. `.env` holds only
  ports and the torch build.
- **Pinned versions.** Base image `python:3.11.17-slim-trixie`, torch 2.14.1, every package pinned as in
  `requirements.txt`; MLflow's image tag matches its pinned version.
- **Non-root** in frontend, backend and model-service. **Health checks**: backend `/health`, frontend
  `/_stcore/health`, MLflow `/health`.
- **Requirements copied before code**, so the slow torch layer is rebuilt only when the requirements change.
  No multi-stage build: every package installs from wheels, so there are no build tools to leave behind.

## Known gaps

- Not built or run on this Mac (Docker isn't installed here); the compose file was only checked to parse.
- MLflow artifacts: experiments created by the native venv store artifact paths as this machine's absolute
  paths, so the MLflow container shows params and metrics but may not open those artifacts.
- File ownership on Linux: the containers write `runs/` as UID 1000; on a host with another UID, adjust it.
