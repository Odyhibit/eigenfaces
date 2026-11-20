"""Frozen, holdout-safe Phase 7 protocol. No model training in this module."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .model import ROOT

RESULTS = ROOT / "output/phase7_frontal_vq"
DATASET = Path("E:/eigenfaces/img_align_celeba")
CURATION = ROOT / "output/frontal_curation"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def names(path):
    values = Path(path).read_text().splitlines()
    if len(values) != len(set(values)) or any(not n or Path(n).name != n for n in values):
        raise ValueError(f"Invalid filename manifest: {path}")
    return set(values)


def choose_training(strict, original_train, unassigned, forbidden, count=50000, seed=42, exclusions=()):
    """All eligible original training filenames first, seeded fill, sorted output."""
    blocked = set(forbidden) | set(exclusions)
    first = sorted(set(strict) & set(original_train) - blocked)
    pool = sorted((set(strict) & set(unassigned)) - blocked - set(first))
    rng = np.random.default_rng(seed)
    if len(first) > count:
        first = [first[i] for i in rng.choice(len(first), count, replace=False)]
    need = min(max(0, count - len(first)), len(pool))
    fill = [pool[i] for i in rng.choice(len(pool), need, replace=False)]
    return sorted(first + fill), {"original_training": len(first), "unassigned_fill": len(fill),
                                 "eligible_fill_pool": len(pool), "requested": count}


def deterministic_sample(pool, count, seed):
    pool = sorted(set(pool))
    return [pool[i] for i in np.random.default_rng(seed).choice(len(pool), min(count, len(pool)), replace=False)]


def validate_groups(groups, forbidden=(), strict=None):
    sets = {}
    for key in ("train", "validation", "test"):
        values = groups[key]
        if not values or len(values) != len(set(values)):
            raise ValueError("Empty or duplicated split")
        sets[key] = set(values)
        if strict is not None and not sets[key] <= set(strict):
            raise ValueError("Non-strict image in frontal split")
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Split overlap")
    if sets["train"] & set(forbidden):
        raise ValueError("Historical holdout admitted to training")


def audit_holdouts(inventory):
    """Inspect saved real-dataset partitions plus explicit historical evaluation roles.

    Synthetic smoke partitions and curation-only source-image reviews are not
    performance holdouts. Training reconstruction examples remain training data.
    """
    heldout, inspected, evidence = set(), set(), {}
    paths = list((ROOT / "models").rglob("split.json")) + list((ROOT / "output").rglob("*split.json"))
    for path in sorted(paths):
        if path.is_relative_to(RESULTS) or path.is_relative_to(ROOT / "models/vq64_frontal"):
            continue
        doc = json.loads(path.read_text())
        if any(part in ("neural_smoke", "vq_smoke") for part in path.parts):
            # These generated fixtures can reuse six-digit filenames; their
            # recorded temporary dataset root distinguishes them from CelebA.
            if "temp" not in str(doc.get("dataset_root", "")).lower():
                raise ValueError("Unexpected smoke partition provenance")
            continue
        if not all(k in doc for k in ("train", "validation", "test")):
            continue
        all_names = {Path(n).name for k in ("train", "validation", "test") for n in doc[k]}
        if not all_names & inventory:
            continue  # Synthetic filenames (0000.jpg etc.), never CelebA images.
        if not all_names <= inventory:
            raise ValueError(f"Partially missing real dataset partition: {path}")
        groups = {k: [Path(n).name for n in doc[k]] for k in ("train", "validation", "test")}
        validate_groups(groups)
        selected = set(groups["validation"]) | set(groups["test"])
        heldout.update(selected)
        evidence[str(path)] = {"sha256": sha(path), "role": "saved validation/test", "excluded_count": len(selected)}
    explicit = [
        ("neural_evaluation/verification.json", ("selected_images",)),
        ("vq_evaluation/verification.json", ("selected_images",)),
        ("phase4_continuation/protocol.json", ("test_names", "monitor_names", "contact_names")),
        ("phase4_continuation/reproducibility.json", ("fresh_test_names",)),
    ]
    for relative, keys in explicit:
        path = ROOT / "output" / relative
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        selected = {Path(n).name for k in keys for n in doc[k]}
        if not selected <= inventory:
            raise ValueError(f"Missing evaluation filenames: {path}")
        heldout.update(selected)
        testkeys = [k for k in keys if k not in ("monitor_names", "contact_names")]
        inspected.update(Path(n).name for k in testkeys for n in doc[k])
        evidence[str(path)] = {"sha256": sha(path), "roles": keys, "excluded_count": len(selected)}
    path = ROOT / "output/neural_diagnostics/verification.json"
    if path.exists():
        doc = json.loads(path.read_text())
        selected = {Path(n).name for k in ("test", "validation") for n in doc["samples"][k]}
        heldout.update(selected)
        inspected.update(Path(n).name for n in doc["samples"]["test"])
        evidence[str(path)] = {"sha256": sha(path), "roles": ["samples.test", "samples.validation"], "excluded_count": len(selected)}
    return heldout, inspected, evidence


def read_geometry(selected=None):
    """Stream the 116 MB curation table, retaining only needed fields/rows."""
    result = {}
    keep_text = {"filename", "image_path", "manual_decision", "validation_reason", "existing_split"}
    with (CURATION / "all_images.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if selected is not None and row["filename"] not in selected:
                continue
            parsed = {}
            for key, value in row.items():
                if key in keep_text:
                    parsed[key] = value
                elif key in {"width", "height", "frontal_score", "eye_roll_degrees", "nose_center_offset", "mouth_center_offset", "eye_mouth_alignment", "strict_margin", "nose_height_ratio", "mouth_height_ratio"} or key.endswith(("_x", "_y")):
                    parsed[key] = float(value) if value else ""
                elif key in {"image_present", "landmarks_valid", "strict_pass", "moderate_pass", "relaxed_pass"}:
                    parsed[key] = value == "True"
            result[row["filename"]] = parsed
    return result


def prepare(output=RESULTS, dataset=DATASET, count=50000, seed=42):
    from .frontal_review import contact_sheet
    output, dataset = Path(output), Path(dataset)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Phase 7 output already exists; inspect it rather than overwrite")
    output.mkdir(parents=True, exist_ok=True)
    strict = names(CURATION / "frontal_strict.txt")
    original = json.loads((ROOT / "models/vq64/split.json").read_text())
    inventory = {p.name: p.relative_to(dataset).as_posix() for p in sorted(dataset.rglob("*.jpg"))}
    if len(inventory) != sum(1 for _ in dataset.rglob("*.jpg")):
        raise ValueError("Ambiguous duplicate image filenames")
    forbidden, inspected, evidence = audit_holdouts(set(inventory))
    train0 = {Path(n).name for n in original["train"]}
    assigned = {Path(n).name for k in ("train", "validation", "test") for n in original[k]}
    unassigned = set(inventory) - assigned
    train, selection = choose_training(strict, train0, unassigned, forbidden, count, seed)
    groups = {"train": train, "validation": sorted(names(CURATION / "splits/strict_validation.txt")),
              "test": sorted(names(CURATION / "splits/strict_test.txt"))}
    validate_groups(groups, forbidden, strict)
    if not set.union(*(set(v) for v in groups.values())) <= set(inventory):
        raise ValueError("Missing selected image")
    manifest = {"format_version": 1, "seed": seed, "method": "Phase 7 strict frontal; original eligible training then seeded unassigned fill",
                "identity_disjoint": False, "dataset_root": str(dataset),
                **{k: [inventory[n] for n in values] for k, values in groups.items()}}
    for key, values in groups.items():
        (output / f"{key}.txt").write_text("".join(n + "\n" for n in values))
    dump(output / "split.json", manifest)
    (output / "historical_holdouts.txt").write_text("".join(n + "\n" for n in sorted(forbidden)))
    (output / "exclusions.csv").write_text("filename,decision,reason,applied\n039728.jpg,exclude,Proposed upward pitch exclusion NOT approved,False\n187720.jpg,exclude,Proposed downward pitch exclusion NOT approved,False\n")
    # A matched-size unspecialized control: original 50k train, never heldouts.
    control = deterministic_sample(train0 - forbidden, min(count, len(train0 - forbidden)), seed)
    dump(output / "control_split.json", {**manifest, "method": "Matched-size original less-restricted training distribution; never started automatically",
                                        "train": [inventory[n] for n in sorted(control)]})
    (output / "control_train.txt").write_text("".join(n + "\n" for n in sorted(control)))
    pca = json.loads((ROOT / "models/eigenfaces.json").read_text())
    pca_train = {Path(n).name for n in pca["training_files"]}
    geometry = read_geometry(set(train) | set(groups["validation"]) | set(groups["test"]) | {"039728.jpg", "187720.jpg"})
    review = output / "pitch_audit"
    review.mkdir()
    rng = np.random.default_rng(seed)
    subsets = {
        "random_training": deterministic_sample(train, 64, seed),
        "highest_scores": deterministic_sample(sorted(train, key=lambda n: (-geometry[n]["frontal_score"], n))[:1000], 64, seed),
        "near_strict_boundary": deterministic_sample(sorted(train, key=lambda n: (abs(geometry[n]["strict_margin"]-1), n))[:1000], 64, seed),
        "pitch_proxy_extremes": deterministic_sample(sorted(train, key=lambda n: (geometry[n]["nose_height_ratio"], n))[:250], 32, seed) + deterministic_sample(sorted(train, key=lambda n: (-geometry[n]["nose_height_ratio"], n))[:250], 32, seed+1),
        "known_pitch_examples": ["039728.jpg", "187720.jpg"],
    }
    for title, selected in subsets.items():
        contact_sheet(review / f"{title}.png", [geometry[n] for n in selected], title.replace("_", " "))
    contact_sheet(review / "known_pitch_landmarks.png", [geometry[n] for n in subsets["known_pitch_examples"]], "Unapproved pose exclusions", True)
    dump(review / "selections.json", subsets)
    # Freeze primary/secondary evaluation and visual panels before training.
    fresh = set(groups["test"]) - inspected - pca_train
    primary = deterministic_sample(fresh, 1000, seed+1)
    secondary_pool = {Path(n).name for n in original["test"]} - strict - pca_train
    fresh_secondary = secondary_pool - inspected
    secondary = deterministic_sample(fresh_secondary, 1000, seed+2)
    monitor = [manifest["validation"][i] for i in np.random.default_rng(seed+1).choice(len(groups["validation"]), min(20, len(groups["validation"])), replace=False)]
    frontal_pool = deterministic_sample(groups["test"], 250, seed+3)
    from .preprocess import preprocess_image
    from .vq_evaluate import diverse_indices, diversity_features
    originals = np.stack([preprocess_image(dataset / inventory[n]) for n in frontal_pool])
    varied = [frontal_pool[i] for i in diverse_indices(diversity_features(originals), 25)]
    panels = {"random_frontal": deterministic_sample(primary, 25, seed+4), "varied_frontal": varied,
              "borderline_frontal": deterministic_sample(sorted(groups["test"], key=lambda n: (abs(geometry[n]["strict_margin"]-1), n))[:250], 25, seed+5),
              "nonfrontal": deterministic_sample(secondary, 25, seed+6)}
    for category, selected in panels.items():
        if category != "nonfrontal":
            contact_sheet(review / f"preselected_{category}.png", [geometry[n] for n in selected], "Frozen evaluation panel: " + category)
    dump(output / "evaluation_protocol.json", {"seed": seed, "primary_fresh_frontal": primary,
        "complete_frontal": groups["test"], "secondary_fresh_nonfrontal": secondary,
        "monitor_names": monitor, "visual_panels": panels, "varied_candidates": frontal_pool,
        "selection_uses_originals_only": True, "bootstrap_resamples": 5000,
        "previously_inspected_test_count": len(inspected), "fresh_frontal_pool": len(fresh),
        "fresh_nonfrontal_pool": len(fresh_secondary)})
    source = [CURATION / "frontal_strict.txt", CURATION / "configuration.json", CURATION / "all_images.csv",
              CURATION / "splits/strict_train.txt", CURATION / "splits/strict_validation.txt", CURATION / "splits/strict_test.txt",
              ROOT / "models/vq64/best.pt", ROOT / "models/vq64/last.pt", ROOT / "models/vq64/config.json", ROOT / "models/eigenfaces.json"]
    protected = {}
    for root in (ROOT / "models", ROOT / "output"):
        for path in root.rglob("*"):
            if path.is_file() and not path.is_relative_to(output):
                protected[str(path)] = sha(path)
    dump(output / "protected_inputs.json", protected)
    identities = list(Path("E:/eigenfaces").rglob("identity*.txt"))
    identity_audit = {"annotation_files": [str(p) for p in identities], "identity_disjoint": False}
    if identities:
        mapping = dict(line.split() for line in identities[0].read_text().splitlines() if line.strip())
        ids = {k: {mapping[n] for n in v if n in mapping} for k,v in groups.items()}
        identity_audit.update(source_sha256=sha(identities[0]), missing={k: len(set(v)-mapping.keys()) for k,v in groups.items()},
            overlaps={a+"_"+b: len(ids[a]&ids[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))})
    else:
        identity_audit["limitation"] = "No identity annotations available; filename separation cannot establish identity separation."
    metadata = {"seed": seed, "dataset": str(dataset), "counts": {k:len(v) for k,v in groups.items()},
                "selection": selection, "forbidden_filenames": len(forbidden), "holdout_sources": evidence,
                "sources": {str(p): sha(p) for p in source}, "manifests": {k: sha(output/f"{k}.txt") for k in groups},
                "pca_training_overlap": {k: len(set(v)&pca_train) for k,v in groups.items()},
                "pca_training_overlap_filenames": sorted(set(train)&pca_train), "identity_audit": identity_audit,
                "manual_exclusions_applied": [], "unapproved_pose_examples": {n:{"training":n in train,"validation":n in groups['validation'],"test":n in groups['test']} for n in ("039728.jpg","187720.jpg")},
                "known_limitations": ["Five-point geometry does not reliably reject head pitch", "Unapproved manual exclusions left unapplied", "Curation reviews are not performance holdouts"]}
    dump(output / "manifest_metadata.json", metadata)
    return metadata
