"""A readable model catalogue in the bucket, next to MLflow's own run folders.

MLflow keeps each registered package under a random run id (mlflow/<run id>/artifacts/...), which says
nothing to a person browsing the bucket. publish() copies every registered version into one folder per
weights version, named for what it holds, with a model card and Cloud Storage custom metadata on every file:

    <bucket>/models/<model>/
        README.md                       all versions: aliases, PRODUCTION, experiment, kind, weights, date, folder
        catalog.json                    the same for scripts
        aliases/<alias>.json            which version and folder each alias points to (champion = production)
        <experiment>_<single|Nfold>_<weights>/
            MODEL_CARD.md               what it is, training, preprocessing, results it carries, intended use, lineage
            metadata.json, config.yaml, best.pt (or fold0/ ... foldN/)

Every file there carries metadata (model, registry-versions, aliases, production, experiment, kind,
weights-version, registered, members, combination, intended-use), shown under the object's details in the
Cloud Storage console and by `gcloud storage objects describe`. Versions with the same weights share a
folder. Each registry version is tagged catalog_uri, and the cloud build reads that folder.
MLflow's own copies are left as they are. A folder whose weights are already there is not uploaded again;
only its metadata, model card and the index files are rewritten.

The catalogue sits at <parent of the hippo-models artifact store>/models/<model>, e.g. gs://BUCKET/mlflow ->
gs://BUCKET/models/hippo-unet. A local folder works the same way (the tests); its metadata goes to
.object_metadata.json at the catalogue root, since files have no metadata of their own.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from urllib.parse import urlparse

CATALOG = "models"
INTENDED_USE = "Research and education only; not a medical device; not for diagnosis or any clinical decision."


def folder_name(meta: dict) -> str:
    """Folder for a package, e.g. random_trim_5fold_a2d551dad98a, so the bucket says what it holds."""
    members = len(meta["members"])
    return f"{meta['experiment']}_{'single' if members == 1 else f'{members}fold'}_{meta['version']}"


def row_folder(row: dict) -> str | None:
    """The folder a registry row's package would have (folder_name() without downloading it), if it can tell."""
    found = row["kind"].split("-fold")[0]
    members = "single" if row["kind"] == "single run" else f"{found}fold" if found.isdigit() else None
    if not members or row["experiment"] in ("", "?") or not row["weights_version"]:
        return None
    return f"{row['experiment']}_{members}_{row['weights_version']}"


def catalog_root(artifact_location: str, model_name: str) -> str:
    """<parent of the artifact store>/models/<model>: gs://B/mlflow -> gs://B/models/<model> (gs://B -> the same)."""
    location = artifact_location.rstrip("/")
    scheme, _, rest = location.partition("://")
    if not rest:  # a plain local path
        return str(Path(location).parent / CATALOG / model_name)
    head, _, tail = rest.partition("/")
    parent = rest.rsplit("/", 1)[0] if tail else head
    return f"{scheme}://{parent}/{CATALOG}/{model_name}"


class LocalStore:
    """The catalogue in a local folder (file:// or a path); metadata kept in .object_metadata.json."""

    def __init__(self, uri: str):
        self.root = Path(urlparse(uri).path if uri.startswith("file://") else uri)
        self.base = uri.rstrip("/")
        self.index = self.root / ".object_metadata.json"

    def uri(self, rel: str = "") -> str:
        return f"{self.base}/{rel}".rstrip("/")

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def _metadata(self, rel: str, metadata: dict) -> None:
        found = json.loads(self.index.read_text()) if self.index.exists() else {}
        found[rel] = metadata
        self.index.parent.mkdir(parents=True, exist_ok=True)
        self.index.write_text(json.dumps(found, indent=2, sort_keys=True))

    def metadata(self, rel: str) -> dict:
        return json.loads(self.index.read_text()).get(rel, {}) if self.index.exists() else {}

    def upload(self, path: Path, rel: str, metadata: dict) -> None:
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_bytes(Path(path).read_bytes())
        self._metadata(rel, metadata)

    def write_text(self, text: str, rel: str, metadata: dict) -> None:
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_text(text)
        self._metadata(rel, metadata)

    def read_text(self, rel: str) -> str:
        return (self.root / rel).read_text()

    def set_metadata(self, rel: str, metadata: dict) -> None:
        self._metadata(rel, metadata)

    def list(self, prefix: str) -> list[str]:
        return sorted(str(p.relative_to(self.root)) for p in (self.root / prefix).rglob("*") if p.is_file()) \
            if (self.root / prefix).exists() else []

    def delete(self, rel: str) -> None:
        (self.root / rel).unlink(missing_ok=True)


class GcsStore:
    """The catalogue in a Cloud Storage bucket, with custom metadata on each object."""

    def __init__(self, uri: str, client=None):
        from google.cloud import storage  # only needed for gs:// catalogues
        bucket, _, prefix = uri[len("gs://"):].partition("/")
        self.client = client or storage.Client()
        self.bucket = self.client.bucket(bucket)
        self.prefix = prefix.strip("/")
        self.base = uri.rstrip("/")

    def _name(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def uri(self, rel: str = "") -> str:
        return f"{self.base}/{rel}".rstrip("/")

    def exists(self, rel: str) -> bool:
        return self.bucket.blob(self._name(rel)).exists()

    def upload(self, path: Path, rel: str, metadata: dict) -> None:
        blob = self.bucket.blob(self._name(rel))
        blob.metadata = metadata
        blob.upload_from_filename(str(path))

    def write_text(self, text: str, rel: str, metadata: dict) -> None:
        blob = self.bucket.blob(self._name(rel))
        blob.metadata = metadata
        blob.upload_from_string(text, content_type="application/json" if rel.endswith(".json") else "text/markdown")

    def read_text(self, rel: str) -> str:
        return self.bucket.blob(self._name(rel)).download_as_text()

    def set_metadata(self, rel: str, metadata: dict) -> None:
        blob = self.bucket.get_blob(self._name(rel))
        blob.metadata = metadata
        blob.patch()

    def list(self, prefix: str) -> list[str]:
        start = len(self.prefix) + 1 if self.prefix else 0
        return sorted(b.name[start:] for b in self.client.list_blobs(self.bucket, prefix=self._name(prefix)))

    def delete(self, rel: str) -> None:
        self.bucket.blob(self._name(rel)).delete()


def store_for(uri: str):
    return GcsStore(uri) if uri.startswith("gs://") else LocalStore(uri)


def object_metadata(model_name: str, meta: dict, group: list[dict]) -> dict:
    """Cloud Storage custom metadata for every file of one model folder (strings only)."""
    aliases = sorted({a for r in group for a in r["aliases"]})
    return {"model": model_name, "registry-versions": ",".join(str(r["version"]) for r in sorted(group, key=lambda r: r["version"])),
            "aliases": ",".join(aliases) or "-", "production": str(any(r["production"] for r in group)).lower(),
            "experiment": meta["experiment"], "kind": group[0]["kind"], "weights-version": meta["version"],
            "registered": min(r["registered"] for r in group), "members": str(len(meta["members"])),
            "combination": meta["combination"], "intended-use": INTENDED_USE}


def model_card(model_name: str, meta: dict, group: list[dict], files: list[str]) -> str:
    """MODEL_CARD.md for one model folder, from its package metadata and its registry versions."""
    aliases = sorted({a for r in group for a in r["aliases"]})
    production = any(r["production"] for r in group)
    lines = [f"# {meta['name']}: {meta['experiment']}, {group[0]['kind']}", "",
             f"**{model_name}**, weights version `{meta['version']}`" + (" · **PRODUCTION**" if production else ""), "",
             f"> {meta.get('intended_use', INTENDED_USE)}", "",
             "| | |", "|---|---|",
             f"| Registry versions | {', '.join(str(r['version']) for r in sorted(group, key=lambda r: r['version']))} |",
             f"| Aliases | {', '.join(f'`{a}`' for a in aliases) or '-'} |",
             f"| Registered | {min(r['registered'] for r in group)} |",
             f"| Built from | `runs/{meta['experiment']}`, {len(meta['members'])} model(s): {', '.join(meta['members'])} |",
             f"| Combination | {meta['combination']} |",
             f"| Training MLflow runs | {', '.join(str(i) for i in meta.get('mlflow_run_ids', [])) or '-'} |",
             f"| Package created | {meta.get('created', '?')} |", "",
             "## What it does", "",
             "A MONAI 3D U-Net that outlines the hippocampus on a T1-weighted MRI crop and splits it into anterior (1) "
             "and posterior (2); volumes in ml. Served by `hippo.api`"
             + (f" as `?model={aliases[0]}`" if aliases else "") + ".", "",
             f"- **Input:** {meta.get('input', {}).get('contrast', 'T1-weighted')}, "
             f"{meta.get('input', {}).get('voxel_mm', 1.0)} mm voxels, a crop of about "
             f"{' × '.join(str(n) for n in meta.get('input', {}).get('crop_voxels_typical', [35, 50, 35]))} voxels"
             + (", or a whole brain in MNI space (atlas boxes stored)" if "mni_boxes" in meta else "") + ".",
             f"- **Preprocessing:** RAS orientation, intensity `{meta['normalize']}`, padding to a multiple of 8.",
             f"- **Training labels' range:** whole volume {meta['training_label_range']['vol_min']}–"
             f"{meta['training_label_range']['vol_max']} ml, at least {meta['training_label_range']['margin_min']} mm "
             "from the crop edge (the API flags results outside it).", ""]
    cv = meta.get("cross_validation") or []
    lines += ["## Results it carries", ""]
    if cv:
        lines += ["Cross-validation (mean ± SD over the fold models, each on the whole test set):", "",
                  "| Test set | Post-processing | Label | Dice | HD95 (mm) | NSD | ICC |", "|---|---|---|---:|---:|---:|---:|"]
        fmt = lambda r, k, d: f"{r[k + '_mean']:.{d}f} ± {r[k + '_sd']:.{d}f}" if isinstance(r.get(k + "_mean"), float) else "–"
        lines += [f"| {r['test set']} | {r['post-processing']} | {r['label']} | {fmt(r, 'dice', 3)} | {fmt(r, 'hd95', 2)} "
                  f"| {fmt(r, 'nsd', 3)} | {fmt(r, 'icc', 3)} |" for r in cv]
    else:
        lines += [f"None in the package (a single run). See the evaluation of `runs/{meta['experiment']}` in the "
                  "project's docs/results.md and its MLflow run."]
    lines += ["", "## Files", ""] + [f"- `{f}`" for f in files] + [
        "", "Written by hippo.model_catalog. Research use only, not for diagnosis."]
    return "\n".join(lines) + "\n"


def readme(model_name: str, entries: list[dict], production_alias: str) -> str:
    """README.md at the catalogue root: one row per model folder."""
    lines = [f"# {model_name}: model catalogue", "",
             f"One folder per model (weights version). **PRODUCTION** = the alias `{production_alias}`, which the "
             "cloud build deploys. `aliases/<alias>.json` says where each alias points. Research use only, not for "
             "diagnosis.", "",
             "| Folder | Registry versions | Aliases | Experiment | Model | Weights | Registered |",
             "|---|---|---|---|---|---|---|"]
    for e in entries:
        aliases = ", ".join(f"`{a}`" for a in e["aliases"]) + (" **PRODUCTION**" if e["production"] else "")
        lines.append(f"| [{e['folder']}]({e['folder']}/MODEL_CARD.md) | {', '.join(map(str, e['versions']))} "
                     f"| {aliases or '-'} | {e['experiment']} | {e['kind']} | `{e['weights_version']}` | {e['registered']} |")
    return "\n".join(lines) + "\n"


def publish(model_name: str, rows: list[dict], artifact_location: str, production_alias: str, download, tag,
            store=None) -> dict:
    """Write the catalogue for these registry rows (register_model.versions()); returns its root and entries.

    download(uri, folder) fetches a version's package (mlflow.artifacts.download_artifacts);
    tag(version, uri) records the catalogue folder on a registry version.
    """
    store = store or store_for(catalog_root(artifact_location, model_name))
    groups = {}
    for r in sorted(rows, key=lambda r: r["version"]):
        groups.setdefault(r["weights_version"], []).append(r)
    entries = []
    for group in groups.values():
        folder = row_folder(group[0])
        if folder and store.exists(f"{folder}/metadata.json"):  # already published: no download, refresh metadata
            meta = json.loads(store.read_text(f"{folder}/metadata.json"))
            files = [rel[len(folder) + 1:] for rel in store.list(f"{folder}/") if not rel.endswith("/MODEL_CARD.md")]
            info = object_metadata(model_name, meta, group)
            for rel in files:
                store.set_metadata(f"{folder}/{rel}", info)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(download(group[0]["uri"], tmp))
                meta = json.loads((local / "metadata.json").read_text())
                folder = folder_name(meta)
                files = sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())
                info = object_metadata(model_name, meta, group)
                for rel in files:
                    store.upload(local / rel, f"{folder}/{rel}", info)
        store.write_text(model_card(model_name, meta, group, files), f"{folder}/MODEL_CARD.md", info)
        for r in group:
            tag(r["version"], store.uri(folder))
        entries.append({"folder": folder, "uri": store.uri(folder), "versions": [r["version"] for r in group],
                        "aliases": sorted({a for r in group for a in r["aliases"]}),
                        "production": any(r["production"] for r in group), "experiment": meta["experiment"],
                        "kind": group[0]["kind"], "weights_version": meta["version"],
                        "registered": min(r["registered"] for r in group), "combination": meta["combination"]})
    entries.sort(key=lambda e: -max(e["versions"]))
    index = {"model": model_name}
    current = {a: e for e in entries for a in e["aliases"]}
    for rel in store.list("aliases/"):  # aliases that moved away or were removed
        if Path(rel).stem not in current:
            store.delete(rel)
    for alias, e in current.items():
        version = max(r["version"] for r in rows if alias in r["aliases"])
        store.write_text(json.dumps({"alias": alias, "version": version, "folder": e["folder"], "uri": e["uri"],
                                     "weights_version": e["weights_version"],
                                     "production": alias == production_alias}, indent=2),
                         f"aliases/{alias}.json", index | {"alias": alias, "folder": e["folder"]})
    store.write_text(readme(model_name, entries, production_alias), "README.md", index)
    store.write_text(json.dumps({"model": model_name, "production_alias": production_alias, "models": entries},
                                indent=2), "catalog.json", index)
    return {"root": store.uri(), "models": entries}
