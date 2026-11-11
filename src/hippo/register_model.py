"""Register a packaged model in MLflow, with its artifacts in a Cloud Storage bucket, for the cloud build.

The cloud reads the model from MLflow's artifact store, not from this machine: this command packages
the model (hippo.package_model), logs the package as the artifacts of a run in the MLflow experiment
"hippo-models", whose artifact store is a bucket (gs://...), and registers that run as a new version of
the registered model (default "hippo-unet"), optionally with an alias such as "champion". Cloud Build
then downloads the version's artifacts into the image (deploy/cloudbuild.yaml, _MODEL_URI).

The tracking data and the registry stay in runs/mlflow.db on this machine (back it up); only the
model artifacts are in the bucket. The training runs keep their own artifact store. Each version is
tagged with the weights version (the hash in metadata.json) and the MLflow run ids of its training runs.

Finding versions in the bucket: run folders have MLflow's random ids, so each new package goes into a readable
folder inside its run (<experiment>_<single|Nfold>_<weights>, e.g. random_trim_5fold_5bf19bf02440), and every
register or alias change rewrites REGISTRY.md and registry.json at the root of the artifact store: one row per
version with its aliases, experiment, single run or ensemble, weights and folder. Production is the version
with the alias "champion" (what deploy/cloudbuild.yaml builds); --list --live URL also asks the running API
which weights it serves, in case the alias moved without a new deployment.

Writing to gs:// needs the google-cloud-storage package and Google credentials on this machine
(gcloud auth application-default login). Any MLflow artifact URI works the same way (a local folder in
the tests).

Usage:
    python -m hippo.register_model --experiment runs/baseline --src data/mri_dataset --single \
        --artifact-root gs://BUCKET/mlflow --alias champion          # first time: creates hippo-models
    python -m hippo.register_model --experiment runs/baseline --src data/mri_dataset --alias champion
    python -m hippo.register_model --show champion                   # artifact URI of that version
    python -m hippo.register_model --show champion --field weights_version   # its weights version (image tag)
    python -m hippo.register_model --promote 1 --alias champion      # rollback: point the alias at version 1
    python -m hippo.register_model --list                            # all versions, aliases, production
    python -m hippo.register_model --list --live https://SERVICE-URL # + which weights Cloud Run serves (HIPPO_API_KEY)
    python -m hippo.register_model --write-index                     # rewrite REGISTRY.md / registry.json in the bucket
    python -m hippo.register_model --build-list champion random_trim ensemble   # several models for one image
    python -m hippo.register_model --publish-catalog                 # (re)write the readable catalogue

--build-list prints name=uri=weights_version entries joined by ";", the cloud build's _MODELS: one image holds
all of them, the first is the default, and hippo.api serves the others as ?model=<name>.

Readable catalogue (hippo.model_catalog): register, --promote and --publish-catalog also write
<bucket>/models/<model>/ with one folder per weights version (e.g. random_trim_5fold_a2d551dad98a), a
MODEL_CARD.md in each, README.md, catalog.json, aliases/<alias>.json, and Cloud Storage custom metadata on
every file. Each registry version is tagged catalog_uri; --show and --build-list then point to the catalogue
(the build reads it), and --show ALIAS --field source still gives MLflow's own copy.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

from . import model_catalog, package_model, tracking
from .model_catalog import folder_name

EXPERIMENT = "hippo-models"
MODEL_NAME = "hippo-unet"
PACKAGE = "model_package"  # temporary folder name; the artifact folder gets a readable name (folder_name)
PRODUCTION = "champion"    # the alias deploy/cloudbuild.yaml builds from, i.e. the model meant for production
INDEX = ("REGISTRY.md", "registry.json")  # written to the root of the artifact store, next to the run folders


def kind(members: int) -> str:
    """'single run' or 'N-fold ensemble'."""
    return "single run" if int(members) == 1 else f"{members}-fold ensemble"


def bucket_exists(name: str) -> bool:
    """Whether a Cloud Storage bucket exists and these credentials can see it."""
    from google.cloud import storage  # only needed for gs:// stores
    return storage.Client().bucket(name).exists()


def check_store(uri: str) -> None:
    """Stop before MLflow writes anything if a gs:// artifact store's bucket isn't there.

    Otherwise a typo or a leftover placeholder only shows up as a failed upload, after the experiment and a
    run have already been created.
    """
    if not uri.startswith("gs://"):
        return
    name = uri[len("gs://"):].split("/")[0]
    try:
        found = bucket_exists(name)
    except Exception as e:  # no credentials, no project, no network: say so plainly
        raise SystemExit(f"Can't check the bucket gs://{name}: {e}") from None
    if not found:
        raise SystemExit(f"The bucket gs://{name} doesn't exist (or this account can't see it). Create it first: "
                         f"gcloud storage buckets create gs://{name} --location=europe-west1")


def model_experiment(artifact_root: str | None) -> str:
    """The id of the MLflow experiment that holds registered model packages, created on first use.

    An experiment's artifact store is fixed when it is created, so the bucket is only needed the first time.
    """
    mlflow = tracking.mlflow
    found = mlflow.get_experiment_by_name(EXPERIMENT)
    if found is None:
        if not artifact_root:
            raise SystemExit(f"MLflow experiment {EXPERIMENT!r} doesn't exist yet: give --artifact-root gs://BUCKET/mlflow")
        check_store(artifact_root)
        return mlflow.create_experiment(EXPERIMENT, artifact_location=artifact_root)
    if artifact_root and found.artifact_location.rstrip("/") != artifact_root.rstrip("/"):
        raise SystemExit(f"{EXPERIMENT!r} already stores its artifacts at {found.artifact_location}, not "
                         f"{artifact_root}; leave out --artifact-root, or use another MLflow database")
    check_store(found.artifact_location)
    return found.experiment_id


def register(experiment: str, src: str | None = None, single: bool = False, name: str = "3D U-Net",
             model_name: str = MODEL_NAME, alias: str | None = None, artifact_root: str | None = None,
             root: str = "runs") -> dict:
    """Package, log, register and publish to the catalogue; returns the version, its weights version, its MLflow
    artifact URI (uri) and its catalogue folder (catalog_uri)."""
    mlflow = tracking.mlflow
    tracking.setup(Path(root))  # the same runs/mlflow.db as training
    experiment_id = model_experiment(artifact_root)
    with tempfile.TemporaryDirectory() as folder:
        meta = package_model.main(experiment, src, str(Path(folder) / PACKAGE), name, single)
        what = kind(len(meta["members"]))
        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=f"{meta['experiment']} {what} {meta['version']}") as run:
            mlflow.log_params({"experiment": meta["experiment"], "members": len(meta["members"]),
                               "combination": meta["combination"], "normalize": meta["normalize"],
                               "mni_endpoint": "mni_boxes" in meta})
            mlflow.set_tags({"weights_version": meta["version"],
                             "training_runs": ",".join(str(i) for i in meta["mlflow_run_ids"])})
            mlflow.log_artifacts(str(Path(folder) / PACKAGE), artifact_path=folder_name(meta))
            uri = mlflow.get_artifact_uri(folder_name(meta))
    client = mlflow.MlflowClient()
    if not client.search_registered_models(filter_string=f"name = '{model_name}'"):
        client.create_registered_model(model_name, description="Hippocampus segmentation (research use only); "
                                                               "artifacts are hippo.package_model packages")
    version = client.create_model_version(model_name, source=uri, run_id=run.info.run_id,
                                          tags={"weights_version": meta["version"], "experiment": meta["experiment"],
                                                "kind": what},
                                          description=f"{name}, {what} ({meta['combination']}), "
                                                      f"from runs/{meta['experiment']}, weights {meta['version']}")
    if alias:
        client.set_registered_model_alias(model_name, alias, version.version)
    print(f"Registered {model_name} version {version.version} (weights {meta['version']})"
          + (f", alias {alias!r}" if alias else "") + f"\nArtifacts: {uri}")
    write_index(model_name, root)
    catalog = publish_catalog(model_name, root) + "/" + folder_name(meta)
    return {"model": model_name, "version": int(version.version), "weights_version": meta["version"], "uri": uri,
            "catalog_uri": catalog}


def versions(model_name: str = MODEL_NAME, root: str = "runs") -> list[dict]:
    """Every version, newest first: aliases, production mark, what it was built from, weights and bucket folder.

    Read from the registry and the versions' runs, so versions registered before the readable tags existed are
    described too.
    """
    tracking.setup(Path(root))
    client = tracking.mlflow.MlflowClient()
    try:
        aliases = client.get_registered_model(model_name).aliases
    except Exception:
        raise SystemExit(f"No registered model {model_name!r} in {root}/mlflow.db") from None
    rows = []
    for v in client.search_model_versions(f"name = '{model_name}'"):
        params = client.get_run(v.run_id).data.params if v.run_id else {}
        names = sorted(a for a, n in aliases.items() if str(n) == str(v.version))
        rows.append({"version": int(v.version), "aliases": names, "production": PRODUCTION in names,
                     "experiment": v.tags.get("experiment", params.get("experiment", "?")),
                     "kind": v.tags.get("kind", kind(params.get("members", 1))),
                     "weights_version": v.tags.get("weights_version", ""),
                     "registered": datetime.fromtimestamp(v.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M"),
                     "description": v.description or "", "uri": v.source})
    return sorted(rows, key=lambda r: -r["version"])


def live_weights(url: str, api_key: str | None = None) -> str:
    """The weights version a running API serves (GET /model), to check what is really in production."""
    request = urllib.request.Request(url.rstrip("/") + "/model", headers={"X-API-Key": api_key} if api_key else {})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)["version"]
    except Exception as e:
        raise SystemExit(f"Can't read {url}/model: {e} (set HIPPO_API_KEY if the service needs a key)") from None


def table(rows: list[dict], live: str | None = None) -> str:
    """The versions as a plain-text table with the bucket folder of each; PRODUCTION marks the champion."""
    lines = [f"{'ver':>3}  {'aliases':<28} {'experiment':<14} {'model':<17} {'weights':<13} registered"]
    for r in rows:
        marks = r["aliases"] + (["PRODUCTION"] if r["production"] else [])
        if live is not None and r["weights_version"] == live:
            marks.append("LIVE")
        lines.append(f"{r['version']:>3}  {', '.join(marks) or '-':<28} {r['experiment']:<14} {r['kind']:<17} "
                     f"{r['weights_version']:<13} {r['registered']}")
    lines.append("")
    lines += [f"v{r['version']}: {r['uri']}" for r in rows]
    return "\n".join(lines)


def markdown(rows: list[dict], model_name: str) -> str:
    """REGISTRY.md: the same table for someone browsing the bucket."""
    out = [f"# {model_name}: registered versions", "",
           f"Written by hippo.register_model at every register or alias change. **Production** = the version with "
           f"the alias `{PRODUCTION}`, which deploy/cloudbuild.yaml builds. Research use only, not for diagnosis.", "",
           "| Version | Aliases | Experiment | Model | Weights | Registered | Folder |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        aliases = ", ".join(f"`{a}`" for a in r["aliases"]) + (" **PRODUCTION**" if r["production"] else "")
        out.append(f"| {r['version']} | {aliases or '-'} | {r['experiment']} | {r['kind']} | `{r['weights_version']}` "
                   f"| {r['registered']} | {r['uri']} |")
    return "\n".join(out) + "\n"


def write_index(model_name: str = MODEL_NAME, root: str = "runs") -> str:
    """Write REGISTRY.md and registry.json to the root of the hippo-models artifact store (the bucket)."""
    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
    rows = versions(model_name, root)
    store = tracking.mlflow.get_experiment_by_name(EXPERIMENT).artifact_location
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / INDEX[0]).write_text(markdown(rows, model_name))
        (Path(folder) / INDEX[1]).write_text(json.dumps({"model": model_name, "production_alias": PRODUCTION,
                                                         "versions": rows}, indent=2))
        repo = get_artifact_repository(store)
        for name in INDEX:
            repo.log_artifact(str(Path(folder) / name))
    return store


def publish_catalog(model_name: str = MODEL_NAME, root: str = "runs") -> str:
    """Write the readable catalogue for every version (hippo.model_catalog) and tag each version catalog_uri."""
    from mlflow.artifacts import download_artifacts
    rows = versions(model_name, root)
    client = tracking.mlflow.MlflowClient()
    store = tracking.mlflow.get_experiment_by_name(EXPERIMENT).artifact_location
    done = model_catalog.publish(
        model_name, rows, store, PRODUCTION,
        download=lambda uri, folder: download_artifacts(artifact_uri=uri, dst_path=folder),
        tag=lambda version, uri: client.set_model_version_tag(model_name, str(version), "catalog_uri", uri))
    print(f"Catalogue: {done['root']} ({len(done['models'])} models: "
          + ", ".join(e["folder"] for e in done["models"]) + ")")
    return done["root"]


def show(alias: str, model_name: str = MODEL_NAME, field: str = "uri", root: str = "runs") -> str:
    """The catalogue folder (else MLflow's copy), MLflow's copy (source), version or weights of an alias's version."""
    tracking.setup(Path(root))
    try:
        version = tracking.mlflow.MlflowClient().get_model_version_by_alias(model_name, alias)
    except Exception:
        raise SystemExit(f"No version of {model_name!r} has the alias {alias!r}") from None
    return str({"uri": version.tags.get("catalog_uri", version.source), "source": version.source,
                "version": version.version, "weights_version": version.tags.get("weights_version", "")}[field])


def build_list(aliases: list[str], model_name: str = MODEL_NAME, root: str = "runs") -> str:
    """The _MODELS value for deploy/cloudbuild.yaml: alias=uri=weights_version per alias, joined by ";"."""
    entries = []
    for alias in aliases:
        if not alias.replace("_", "").replace("-", "").isalnum():
            raise SystemExit(f"{alias!r}: use letters, digits, - and _ only (it becomes a folder and a URL parameter)")
        uri, weights = show(alias, model_name, "uri", root), show(alias, model_name, "weights_version", root)
        if ";" in uri or "=" in uri:
            raise SystemExit(f"The artifact URI of {alias!r} contains ';' or '=': {uri}")
        entries.append(f"{alias}={uri}={weights}")
    return ";".join(entries)


def promote(version: int, alias: str, model_name: str = MODEL_NAME, root: str = "runs") -> None:
    """Point an alias at an existing version (to roll back, or to deploy an older model)."""
    tracking.setup(Path(root))
    client = tracking.mlflow.MlflowClient()
    try:
        found = client.get_model_version(model_name, str(version))
    except Exception:
        raise SystemExit(f"{model_name!r} has no version {version}") from None
    client.set_registered_model_alias(model_name, alias, found.version)
    print(f"{model_name} alias {alias!r} -> version {found.version} (weights {found.tags.get('weights_version', '?')})")
    write_index(model_name, root)
    publish_catalog(model_name, root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Register a packaged model in MLflow (artifacts in a bucket).")
    p.add_argument("--experiment", help="Experiment folder with fold*/ or a single run, e.g. runs/baseline")
    p.add_argument("--src", help="MNI-HISUB25 folder with the SPAM atlas maps (for the whole-brain endpoint)")
    p.add_argument("--single", action="store_true", help="Package the single run, not its folds")
    p.add_argument("--name", default="3D U-Net", help="Model name shown by the API")
    p.add_argument("--model-name", default=MODEL_NAME, help="Registered model name in MLflow")
    p.add_argument("--alias", help="Point this alias at the new version, e.g. champion")
    p.add_argument("--artifact-root", help=f"Artifact store for the {EXPERIMENT!r} experiment, e.g. gs://BUCKET/mlflow "
                                           "(only needed the first time)")
    p.add_argument("--root", default="runs", help="Folder holding mlflow.db")
    p.add_argument("--show", metavar="ALIAS", help="Print the artifact URI of the version holding this alias")
    p.add_argument("--promote", type=int, metavar="VERSION", help="Point --alias at this existing version")
    p.add_argument("--field", default="uri", choices=["uri", "source", "version", "weights_version"],
                   help="With --show: what to print (uri = catalogue folder; source = MLflow's copy; "
                        "weights_version is the image tag)")
    p.add_argument("--publish-catalog", action="store_true",
                   help="(Re)write the readable catalogue <bucket>/models/<model>/ for every version")
    p.add_argument("--list", action="store_true", help="Print every version with its aliases; PRODUCTION = champion")
    p.add_argument("--live", metavar="URL", help="With --list: mark the version the running API serves (LIVE)")
    p.add_argument("--write-index", action="store_true",
                   help=f"Rewrite {' and '.join(INDEX)} at the root of the artifact store (the bucket)")
    p.add_argument("--build-list", nargs="+", metavar="ALIAS",
                   help="Print the cloud build's _MODELS value for these aliases (the first is the default)")
    a = p.parse_args()
    if a.publish_catalog:
        publish_catalog(a.model_name, a.root)
    elif a.build_list:
        print(build_list(a.build_list, a.model_name, a.root))  # only the value, so scripts can use it
    elif a.list:
        live = live_weights(a.live, os.environ.get("HIPPO_API_KEY")) if a.live else None
        print(table(versions(a.model_name, a.root), live))
        if live is not None:
            print(f"\nThe service at {a.live} serves weights {live}.")
    elif a.write_index:
        print(f"Wrote {' and '.join(INDEX)} to {write_index(a.model_name, a.root)}")
    elif a.promote:
        if not a.alias:
            p.error("--promote needs --alias, e.g. --alias champion")
        promote(a.promote, a.alias, a.model_name, a.root)
    elif a.show:
        print(show(a.show, a.model_name, a.field, a.root))  # only the value, so scripts can use it
    elif a.experiment:
        register(a.experiment, a.src, a.single, a.name, a.model_name, a.alias, a.artifact_root, a.root)
    else:
        p.error("give --experiment (to register), --show ALIAS, --list or --write-index")
