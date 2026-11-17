"""HandsOn leaderboard protocol ``handson-eval-v1``.

Public trial lists, hidden verification seeds, scoring and submission checks.
This module uses only the standard library, PyYAML and the lightweight task
registry, so submissions can be checked in CI without Isaac Sim.
"""
import hashlib
import hmac
import json
import math
import os
import re
import subprocess
from pathlib import Path

PROTOCOL = "handson-eval-v1"
SEED_MASK = 0x7FFFFFFF
MAX_PUBLIC_TRIALS = 100
SUITE_SIZES = {"core": 50, "transfer": 16}
SUITE_LABELS = {"core": "Core-50", "transfer": "Transfer-16"}
TRACKS = {
    "multi_task": dict(
        label="Multi-task policy", trials=dict(core=100, transfer=50), ranked_overall=True,
        description="One checkpoint and one inference procedure for all 66 tasks."),
    "single_task": dict(
        label="Single-task policies", trials=dict(core=100), ranked_overall=False,
        description="One checkpoint per Core-50 task; reported on Core-50 only."),
    "zero_shot": dict(
        label="Zero-shot controller", trials=dict(core=10, transfer=10), ranked_overall=True,
        description="No HandsOn demonstrations or finetuning, e.g. prompted foundation-model agents."),
}
TRANSPORTS = ("openpi_websocket", "http_physical_eef52")
MODEL_TYPES = ("VLA", "WAM", "DP", "IL", "VLM", "LLM", "Agent", "RL", "Other")
# Digest of every public (task, trial, seed) triple, 66 tasks x 100 trials.
# Track lists are prefixes of this universe.
PUBLIC_UNIVERSE_SHA256 = "b374ad0c2d0bfae1d93f8ecf8034c948f3c5a97fa4edda3aa15c96a91f9e464d"
# Files whose contents define seeds, settings, language, budgets and cameras.
# Changing any of them requires a new protocol version.
PROTOCOL_FILE_SHA256 = {
    "evaluation/budgets.json": "42673e672a186b071c265ba42b2a63cb33d89579ec03e668e0090c193ee85076",
    "evaluation/settings.json": "beaf08faf21ccba553d8524c65e97399ab17ec6baa8436a5d1df969e07b2cc88",
    "evaluation/camera_contract.json": "a12ce4790c60c60513518c307f467e4abba1cccd209a2934001a2864ecc722ed",
    "evaluation/paper_camera_contracts.json": "fdf1da4cb13df402ef7b7256d9447d13ebf6dfb33f20c6ebfcf0f088c8aa76bc",
    "tasks/assets/goals.json": "a769561b5ef20481ed6897cffefb1a3508f0b4f90a70693975667b695c75de79",
    "tasks/assets/language_bank.json": "5d6d52c55c037c7eaa396e5a490b5d65bf8cda778f6d801ad8b62c46b8b83a2d",
    "tasks/assets/native_contracts.json": "a9089a74840580c0bbbc6b5048affaaead08a2ec557d4b39975582f0ce27e965",
    "tasks/assets/task_checkpoint_names.json": "3cbea3052dfa78917528932f002a813e4cf05c51a6e582b4fa7ac341f98204e6",
}
PACKAGE = Path(__file__).resolve().parents[1]
SUBMISSION_ID = re.compile(r"^[0-9]{8}_[a-z0-9][a-z0-9-]*_[a-z0-9][a-z0-9.-]*$")
ROW_FIELDS = ("task", "suite", "trial", "seed", "status", "success", "reason", "attempts",
              "scored_attempt", "stop_reason", "steps", "goal_id", "prompt_variant", "setting_id",
              "execution", "result_sha256")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------- trial lists
def catalog():
    """Return sorted (task, suite) pairs for the 66 released tasks."""
    from ..tasks.registry import TASKS
    rows = sorted((name, spec.split) for name, spec in TASKS.items())
    counts = {suite: sum(1 for _, split in rows if split == suite) for suite in SUITE_SIZES}
    if counts != SUITE_SIZES:
        raise ValueError(f"Task catalog has {counts}, expected {SUITE_SIZES}")
    return rows


def _seed(digest):
    return int.from_bytes(digest[:4], "big") & SEED_MASK


def public_seed(task, trial):
    payload = _canonical([PROTOCOL, "public", task, int(trial)]).encode()
    return _seed(hashlib.sha256(payload).digest())


def hidden_seeds(key, task, count):
    """Seeds known only to holders of ``key``; never equal to a public seed."""
    if not key:
        raise ValueError("Hidden seeds require a non-empty key")
    public = {public_seed(task, i) for i in range(MAX_PUBLIC_TRIALS)}
    seeds, counter = [], 0
    while len(seeds) < count:
        payload = _canonical([PROTOCOL, "hidden", task, counter]).encode()
        seed = _seed(hmac.new(key, payload, hashlib.sha256).digest())
        counter += 1
        if seed not in public and seed not in seeds:
            seeds.append(seed)
    return seeds


def public_universe_sha256():
    lines = [_canonical(dict(task=task, trial=i, seed=public_seed(task, i)))
             for task, _ in catalog() for i in range(MAX_PUBLIC_TRIALS)]
    return sha256_text("\n".join(lines))


def verify_public_universe():
    actual = public_universe_sha256()
    if actual != PUBLIC_UNIVERSE_SHA256:
        raise ValueError(f"Public {PROTOCOL} trials changed ({actual}); task names or seeding differ")


def manifest(track, *, hidden_key=None, trials=None):
    """Planned trials for a track: public prefixes, or hidden seeds with a key."""
    if track not in TRACKS:
        raise ValueError(f"Unknown track {track!r}; choose one of {sorted(TRACKS)}")
    verify_public_universe()
    plan = TRACKS[track]["trials"]
    rows = []
    for task, suite in catalog():
        if suite not in plan:
            continue
        count = int(trials or plan[suite])
        if hidden_key is None:
            if count > MAX_PUBLIC_TRIALS:
                raise ValueError(f"At most {MAX_PUBLIC_TRIALS} public trials per task")
            seeds = [public_seed(task, i) for i in range(count)]
        else:
            seeds = hidden_seeds(hidden_key, task, count)
        rows.extend(dict(protocol=PROTOCOL, track=track, set="public" if hidden_key is None else "hidden",
                         task=task, suite=suite, trial=i, seed=seed) for i, seed in enumerate(seeds))
    return rows


def manifest_sha256(rows):
    return sha256_text("\n".join(_canonical(row) for row in rows))


# ------------------------------------------------------------ evaluator identity
SOURCE_MANIFEST = "HANDSON_SOURCE.json"
SOURCE_SCOPE = "handson"


def protocol_file_hashes(package=None):
    package = Path(package or PACKAGE)
    return {rel: hashlib.sha256((package / rel).read_bytes()).hexdigest() for rel in PROTOCOL_FILE_SHA256}


def _blob_sha1(path):
    """Git blob id of a working-tree file (symlinks hash their target, as in git)."""
    data = os.readlink(path).encode() if os.path.islink(path) else Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _run_git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def _git_checkout(root):
    """Commit and dirty flag when ``root`` itself is a git checkout and git is on PATH."""
    try:
        top, commit = _run_git(root, "rev-parse", "--show-toplevel", "HEAD").splitlines()
        if Path(top).resolve() != Path(root).resolve():
            return None  # an exported tree inside some other repository
        status = _run_git(root, "status", "--porcelain", "--untracked-files=no", "--", SOURCE_SCOPE)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    return commit, bool(status.strip())


def source_manifest(root, commit="HEAD"):
    """Blob ids of every tracked file under handson/ at ``commit`` (requires git)."""
    full = _run_git(root, "rev-parse", f"{commit}^{{commit}}").strip()
    files = {}
    for entry in _run_git(root, "ls-tree", "-r", "-z", full, "--", SOURCE_SCOPE).split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind != "blob":
            raise ValueError(f"{path}: submodules are not supported in an exported evaluator")
        files[path] = sha
    return dict(schema="handson_source_manifest_v1", protocol=PROTOCOL, git_commit=full, scope=SOURCE_SCOPE + "/",
                files_sha256=sha256_text(_canonical(files)), files=files)


def verify_source_manifest(root):
    """Compare an exported tree with its HANDSON_SOURCE.json; return (commit, mismatched paths)."""
    root = Path(root)
    data = json.loads((root / SOURCE_MANIFEST).read_text())
    files = data.get("files")
    if (data.get("schema") != "handson_source_manifest_v1" or not isinstance(files, dict)
            or data.get("files_sha256") != sha256_text(_canonical(files))
            or not re.fullmatch(r"[0-9a-f]{40}", str(data.get("git_commit")))):
        raise ValueError(f"{root / SOURCE_MANIFEST} is not a valid source manifest")
    mismatched = [path for path, sha in sorted(files.items())
                  if not (root / path).is_file() or _blob_sha1(root / path) != sha]
    return data["git_commit"], mismatched


def identify_source(root=None):
    """Find the evaluator commit: from git, or from a verified HANDSON_SOURCE.json export.

    Cluster nodes often have no git binary. ``suite export-source`` packs a commit with
    a manifest of blob ids; every listed file is re-hashed here, so ``git_dirty`` has
    the same meaning as ``git status`` on the tracked files under handson/.
    """
    root = Path(root or PACKAGE.parents[1])
    checkout = _git_checkout(root)
    if checkout is not None:
        return dict(git_commit=checkout[0], git_dirty=checkout[1], check="git", mismatched=[])
    if (root / SOURCE_MANIFEST).is_file():
        commit, mismatched = verify_source_manifest(root)
        return dict(git_commit=commit, git_dirty=bool(mismatched), check="source_manifest", mismatched=mismatched)
    return dict(git_commit=None, git_dirty=None, check=None, mismatched=[])


def evaluator_identity(source=None):
    """Git commit and protocol-file hashes of the evaluator that runs the trials."""
    source = source or identify_source()
    return dict(protocol=PROTOCOL, git_commit=source["git_commit"], git_dirty=source["git_dirty"],
                protocol_files=protocol_file_hashes())


def export_source(output, commit="HEAD", root=None):
    """Write ``<output>.tar.gz``: ``git archive`` of ``commit`` plus HANDSON_SOURCE.json."""
    import gzip
    import shutil
    import tarfile
    import tempfile
    root = Path(root or PACKAGE.parents[1])
    data = source_manifest(root, commit)
    prefix = f"handson-{data['git_commit'][:12]}"
    output = Path(output)
    with tempfile.TemporaryDirectory() as scratch:
        tar_path = Path(scratch) / "source.tar"
        subprocess.run(["git", "-C", str(root), "archive", "--format=tar", f"--prefix={prefix}/",
                        "-o", str(tar_path), data["git_commit"]], check=True)
        manifest_path = Path(scratch) / SOURCE_MANIFEST
        manifest_path.write_text(json.dumps(data, indent=1) + "\n")
        with tarfile.open(tar_path, "a") as archive:
            info = archive.gettarinfo(str(manifest_path), arcname=f"{prefix}/{SOURCE_MANIFEST}")
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mode = 0o644
            with open(manifest_path, "rb") as handle:
                archive.addfile(info, handle)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(tar_path, "rb") as source, gzip.open(output, "wb", compresslevel=1) as target:
            shutil.copyfileobj(source, target, 1 << 20)
    return dict(output=str(output), directory=prefix, git_commit=data["git_commit"], files=len(data["files"]),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest())


# ---------------------------------------------------------------- run results
def trial_dir(run, task, trial):
    return Path(run) / "trials" / task / f"{int(trial):03d}"


def attempt_dirs(run, task, trial):
    folder = trial_dir(run, task, trial)
    return sorted(p for p in folder.glob("attempt_[0-9][0-9]") if p.is_dir()) if folder.is_dir() else []


def attempt_complete(path):
    """An attempt directory holds the runner's console log and the evaluator's ``episode`` output."""
    episode = path / "episode"
    return (episode / "RESULT.json").is_file() and not (episode / "ERROR.json").exists()


def _load(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def _execution(job):
    probe = job.get("policy_probe", {})
    return dict(transport=probe.get("transport"), action_horizon=probe.get("action_horizon"),
                history_frames=probe.get("history_frames", 1), history_stride=probe.get("history_stride"),
                render_stride=job.get("benchmark", {}).get("render_stride"))


def collect_row(run, planned):
    """One results.jsonl row: the first completed attempt is the scored one."""
    task, trial, seed = planned["task"], planned["trial"], planned["seed"]
    row = dict(task=task, suite=planned["suite"], trial=trial, seed=seed, status="missing", success=None,
               reason="no completed attempt", attempts=0, scored_attempt=None, stop_reason=None, steps=None,
               goal_id=None, prompt_variant=None, setting_id=None, execution=None, result_sha256=None)
    attempts = attempt_dirs(run, task, trial)
    row["attempts"] = len(attempts)
    done = next((p for p in attempts if attempt_complete(p)), None)
    if done is None:
        return row
    episode = done / "episode"
    raw = (episode / "RESULT.json").read_bytes()
    result, job = json.loads(raw), _load(episode / "JOB.json")
    prompt, setting = _load(episode / "PROMPT.json"), _load(episode / "SETTING_SELECTION.json")
    checks = [
        (result.get("task") == task, "result task differs from the planned trial"),
        (result.get("seed") == seed, "result seed differs from the planned trial"),
        (result.get("protocol") == "handson_native_eef52", "not a native EEF52 evaluation"),
        (result.get("physics_backend") == "physx", "not the PhysX backend"),
        (result.get("paper_physics_protocol") is True, "paper physics protocol disabled"),
        (result.get("diagnostic_only") is False, "diagnostic run, not a policy rollout"),
        (result.get("counts_as_formal_trial") is True, "not a formal trial (bounded or diagnostic run)"),
        (result.get("complete_evaluation_episode") is True, "episode did not complete"),
        (isinstance(result.get("success"), bool), "no native success outcome"),
        (result.get("paper_variant") is None, "historical paper variant selected"),
    ]
    failed = [message for ok, message in checks if not ok]
    row.update(scored_attempt=done.name, stop_reason=result.get("stop_reason"), steps=result.get("steps"),
               goal_id=prompt.get("goal_id"), prompt_variant=prompt.get("prompt_variant"),
               setting_id=setting.get("setting_id"), execution=_execution(job),
               result_sha256=hashlib.sha256(raw).hexdigest())
    if failed:
        row.update(status="invalid", reason="; ".join(failed))
    else:
        row.update(status="scored", success=result["success"], reason=None)
    return row


def results_text(rows):
    return "".join(_canonical({key: row[key] for key in ROW_FIELDS}) + "\n" for row in rows)


# -------------------------------------------------------------------- scoring
def _rate(successes, trials):
    return round(successes / trials, 6) if trials else None


def _macro(task_rows):
    if not task_rows:
        return None
    rates = [row["successes"] / row["planned"] for row in task_rows]
    mean = sum(rates) / len(rates)
    variance = sum(p * (1 - p) / row["planned"] for p, row in zip(rates, task_rows)) / len(rates) ** 2
    half = 1.96 * math.sqrt(variance)
    return dict(tasks=len(task_rows), planned=sum(r["planned"] for r in task_rows),
                successes=sum(r["successes"] for r in task_rows), rate=round(mean, 6),
                percent=round(100 * mean, 2), ci95_percent=[round(100 * max(0.0, mean - half), 2),
                                                           round(100 * min(1.0, mean + half), 2)])


def score(rows, track):
    """Task-macro rates. Missing and invalid trials count as failures."""
    tasks = {}
    for row in rows:
        entry = tasks.setdefault(row["task"], dict(task=row["task"], suite=row["suite"], planned=0,
                                                   scored=0, successes=0))
        entry["planned"] += 1
        entry["scored"] += row["status"] == "scored"
        entry["successes"] += row["status"] == "scored" and row["success"] is True
    task_rows = sorted(tasks.values(), key=lambda row: (row["suite"], row["task"]))
    for row in task_rows:
        row["rate"] = _rate(row["successes"], row["planned"])
    suites = {suite: _macro([row for row in task_rows if row["suite"] == suite])
              for suite in TRACKS[track]["trials"]}
    overall = _macro(task_rows) if TRACKS[track]["ranked_overall"] else None
    return dict(suites=suites, overall=overall, tasks=task_rows)


def build_summary(rows, track, evaluator, manifest_set="public", manifest_digest=None):
    statuses = {status: sum(row["status"] == status for row in rows) for status in ("scored", "invalid", "missing")}
    executions = {_canonical(row["execution"]) for row in rows if row["execution"] is not None}
    execution = json.loads(executions.pop()) if len(executions) == 1 else None
    summary = dict(
        schema="handson_submission_summary_v1", protocol=PROTOCOL, track=track, set=manifest_set,
        manifest_sha256=manifest_digest or manifest_sha256(manifest(track)),
        results_sha256=sha256_text(results_text(rows)), evaluator=evaluator,
        execution=execution, consistent_execution=len(executions) == 0 and execution is not None,
        counts=dict(planned=len(rows), **statuses,
                    successes=sum(row["status"] == "scored" and row["success"] is True for row in rows),
                    retried_trials=sum(row["attempts"] > 1 for row in rows),
                    attempts=sum(row["attempts"] for row in rows)),
    )
    summary.update(score(rows, track))
    return summary


def consistency(public, hidden):
    """Compare a submitted public score with a hidden-seed rerun of the same system.

    Returns the decision and the z statistic of the macro-rate difference. The
    rerun is consistent when the difference lies within 1.96 combined standard
    errors (normal approximation of task-macro means)."""
    key = "overall" if public.get("overall") else "suites"
    def stats(summary):
        if key == "overall":
            block = summary["overall"]
            rows = summary["tasks"]
        else:
            block = summary["suites"]["core"]
            rows = [row for row in summary["tasks"] if row["suite"] == "core"]
        rates = [row["successes"] / row["planned"] for row in rows]
        var = sum(p * (1 - p) / row["planned"] for p, row in zip(rates, rows)) / len(rows) ** 2
        return block["rate"], var
    p, vp = stats(public)
    h, vh = stats(hidden)
    se = math.sqrt(vp + vh)
    z = (p - h) / se if se else (0.0 if p == h else math.inf)
    return dict(metric="overall" if key == "overall" else "core", public_rate=p, hidden_rate=h,
                z=round(z, 3), consistent=abs(z) <= 1.96)


# ----------------------------------------------------------------- submission
REQUIRED_METADATA = {
    ("protocol",): str, ("track",): str,
    ("submission", "team"): str, ("submission", "organization"): str, ("submission", "contact_email"): str,
    ("model", "name"): str, ("model", "version"): str, ("model", "type"): list,
    ("links", "artifacts"): str,
    ("training", "handson_data"): str, ("training", "external_data"): str, ("training", "pretrained_init"): str,
    ("training", "transfer16_data"): bool, ("training", "evaluation_feedback_used"): bool,
    ("interface", "transport"): str, ("interface", "action_horizon"): int,
    ("interface", "history_frames"): int, ("interface", "uses_task_identifier"): bool,
}


def _get(data, path):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _is_url(value):
    return isinstance(value, str) and re.match(r"^https?://\S+$", value) is not None


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).returncode == 0


def validate_submission(path, *, git_repo=None, git_ref=None):
    """Return ``dict(errors=[...], warnings=[...])`` for one submission directory."""
    import yaml
    path = Path(path)
    errors, warnings = [], []
    if not SUBMISSION_ID.match(path.name):
        errors.append("directory name must be YYYYMMDD_<team>_<model> in lowercase letters, digits, '-' or '.'")
    files = {name: path / name for name in ("metadata.yaml", "results.jsonl", "summary.json")}
    missing = [name for name, file in files.items() if not file.is_file()]
    if missing:
        return dict(errors=errors + [f"missing {', '.join(missing)}"], warnings=warnings)
    try:
        metadata = yaml.safe_load(files["metadata.yaml"].read_text())
        summary = json.loads(files["summary.json"].read_text())
        text = files["results.jsonl"].read_text()
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (yaml.YAMLError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return dict(errors=errors + [f"unreadable submission file: {exc}"], warnings=warnings)
    if not isinstance(metadata, dict):
        return dict(errors=errors + ["metadata.yaml must be a mapping"], warnings=warnings)
    for key, kind in REQUIRED_METADATA.items():
        value = _get(metadata, key)
        if not isinstance(value, kind) or (kind is str and not value.strip()) or (kind is list and not value):
            errors.append(f"metadata.{'.'.join(key)} must be a non-empty {kind.__name__}")
    track = metadata.get("track")
    if metadata.get("protocol") != PROTOCOL:
        errors.append(f"metadata.protocol must be {PROTOCOL}")
    if track not in TRACKS:
        return dict(errors=errors + [f"metadata.track must be one of {sorted(TRACKS)}"], warnings=warnings)
    if not set(_get(metadata, ("model", "type")) or []) <= set(MODEL_TYPES):
        errors.append(f"metadata.model.type entries must be in {list(MODEL_TYPES)}")
    if _get(metadata, ("training", "transfer16_data")) is not False:
        errors.append("Transfer-16 tasks must not contribute training data (training.transfer16_data: false)")
    if _get(metadata, ("training", "evaluation_feedback_used")) is not False:
        errors.append("evaluation outcomes must not be used to train or tune the system")
    if _get(metadata, ("interface", "uses_task_identifier")) is not False:
        errors.append("the policy must not use the task identifier or seed to choose checkpoints, code or prompts")
    if _get(metadata, ("interface", "transport")) not in TRANSPORTS:
        errors.append(f"metadata.interface.transport must be one of {list(TRANSPORTS)}")
    if track == "zero_shot" and str(_get(metadata, ("training", "handson_data"))).strip().lower() != "none":
        errors.append("zero_shot submissions must not train on HandsOn data (training.handson_data: none)")
    if not _is_url(_get(metadata, ("links", "artifacts"))):
        errors.append("links.artifacts must be an http(s) URL to the raw episode directories")
    for key, purpose in (("report", "a paper or technical report"), ("code", "inference code"),
                         ("checkpoint", "the evaluated weights or API model version")):
        if not _is_url(_get(metadata, ("links", key))):
            warnings.append(f"links.{key} is not a public URL; {purpose} is required for verification")
    # Planned trials and per-trial rows.
    expected = manifest(track)
    plan = {(row["task"], row["trial"]): row for row in expected}
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(ROW_FIELDS):
            errors.append(f"results.jsonl line {index + 1} must have exactly the fields {list(ROW_FIELDS)}")
            return dict(errors=errors, warnings=warnings)
        key = (row["task"], row["trial"])
        if key in seen:
            errors.append(f"duplicate trial {key}")
        seen.add(key)
        planned = plan.get(key)
        if planned is None:
            errors.append(f"trial {key} is not in the public {track} manifest")
        elif row["seed"] != planned["seed"] or row["suite"] != planned["suite"]:
            errors.append(f"trial {key} has seed {row['seed']}, expected {planned['seed']}")
    absent = len(set(plan) - seen)
    if absent:
        errors.append(f"{absent} planned trials have no row")
    order = [(row["task"], row["trial"]) for row in expected]
    if [(row["task"], row["trial"]) for row in rows] != [key for key in order if key in seen]:
        errors.append("rows must follow the manifest order (python -m handson.evaluation.suite summarize)")
    unscored = [row for row in rows if row["status"] != "scored"]
    if unscored:
        errors.append(f"{len(unscored)} trials are missing or invalid; every planned trial needs a scored outcome")
    if text != results_text(rows):
        errors.append("results.jsonl is not in canonical form; regenerate it with the suite summarize command")
    # Evaluator identity and recomputed summary.
    evaluator = summary.get("evaluator") or {}
    if summary.get("protocol") != PROTOCOL or summary.get("track") != track or summary.get("set") != "public":
        errors.append("summary.json protocol, track or set does not match metadata.yaml")
    if evaluator.get("protocol_files") != PROTOCOL_FILE_SHA256:
        errors.append(f"evaluator protocol files differ from {PROTOCOL}")
    commit = evaluator.get("git_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        errors.append("evaluator.git_commit is missing; run the suite from a git checkout of HandsOn")
    elif git_repo is not None:
        if not _git(git_repo, "cat-file", "-e", f"{commit}^{{commit}}"):
            errors.append(f"evaluator commit {commit} is not in the HandsOn repository")
        elif git_ref and not _git(git_repo, "merge-base", "--is-ancestor", commit, git_ref):
            errors.append(f"evaluator commit {commit} is not on {git_ref}")
    if evaluator.get("git_dirty") is not False:
        errors.append("the evaluator checkout had local changes under handson/")
    recomputed = build_summary(rows, track, evaluator) if rows else None
    if recomputed is not None and recomputed != summary:
        changed = sorted(key for key in set(recomputed) | set(summary) if recomputed.get(key) != summary.get(key))
        errors.append(f"summary.json does not match results.jsonl (fields: {', '.join(changed)})")
    execution = summary.get("execution") or {}
    if not summary.get("consistent_execution"):
        errors.append("all trials must use one transport, horizon and history setting")
    for key in ("transport", "action_horizon", "history_frames"):
        if execution.get(key) != _get(metadata, ("interface", key)):
            errors.append(f"metadata.interface.{key} differs from the recorded execution ({execution.get(key)})")
    return dict(errors=errors, warnings=warnings)
