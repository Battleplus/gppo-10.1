"""Copy frozen preparations and preserve every pre-existing staged blob."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

EVIDENCE = Path(__file__).resolve().parent
WORKSPACE = EVIDENCE.parents[1]
REPOSITORY = WORKSPACE / "research-progress-archive"
PACKAGES = (
    EVIDENCE.parent / "w1-remote-runtime-dependency-repair-v1",
    EVIDENCE.parent / "w1-action-conditioned-task-outcome-label-repair-v6-dependency-cpu-settlement",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root):
    return {path.relative_to(root).as_posix(): digest(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def git(*arguments):
    process = subprocess.run(["git", *arguments], cwd=REPOSITORY,
                             capture_output=True, check=True)
    return process.stdout


def index_entries():
    result = {}
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if entry:
            identity, path = entry.split(b"\t", 1)
            result[path.decode("utf-8")] = identity.decode("ascii")
    return result


def main():
    before_index = index_entries()
    staged_before = git("diff", "--cached", "--name-only", "-z").split(b"\0")
    report = {"existing_staged_count": sum(bool(path) for path in staged_before),
              "packages": [], "remote_archived": False,
              "remote_blocker": "Windows ssh-agent Disabled/Stopped; signing requirement retained",
              "commit_created": False, "push_attempted": False}
    destination_root = REPOSITORY / "docs" / "archive" / "w1-v6-runtime-final-delivery-20261002"
    if destination_root.exists():
        raise SystemExit("ARCHIVE_DESTINATION_EXISTS_DO_NOT_OVERWRITE")
    for source in PACKAGES:
        if not (source / "delivery-hashes.json").exists() and not (source / "hashes.json").exists():
            raise SystemExit("SOURCE_NOT_FROZEN")
        source_identity = snapshot(source)
        if any("__pycache__" in Path(path).parts or Path(path).suffix == ".pyc"
               for path in source_identity):
            raise SystemExit("CACHE_IN_FINAL_DELIVERY")
        destination = destination_root / source.name
        shutil.copytree(source, destination)
        copied_identity = snapshot(destination)
        if copied_identity != source_identity:
            raise SystemExit("ARCHIVE_COPY_IDENTITY_MISMATCH")
        selected = []
        for relative in source_identity:
            path = destination / relative
            if path.suffix in {".py", ".md", ".json", ".txt", ".log", ".jsonl"} and path.stat().st_size <= 1024 * 1024:
                if path.suffixes[-2:] != [".tar", ".gz"]:
                    selected.append(path.relative_to(REPOSITORY).as_posix())
        for offset in range(0, len(selected), 40):
            git("add", "--", *selected[offset:offset + 40])
        report["packages"].append({"source": str(source), "archive": str(destination),
                                   "all_files_equal": True, "file_count": len(source_identity),
                                   "staged_small_file_count": len(selected), "files": source_identity})
    evidence_files = [path for path in EVIDENCE.iterdir() if path.is_file()
                      and path.name != "local-archive-status.json"
                      and path.suffix in {".py", ".md", ".json", ".stdout", ".stderr"}]
    evidence_files.extend(path for path in (EVIDENCE / "remote-runtime").iterdir()
                          if path.is_file() and path.suffix in {".json", ".stdout", ".stderr", ".exit", ".sha256"})
    evidence_identities = {}
    for path in evidence_files:
        if path.stat().st_size > 1024 * 1024:
            continue
        relative = path.relative_to(EVIDENCE)
        copied = destination_root / EVIDENCE.name / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copied)
        if digest(copied) != digest(path):
            raise SystemExit("ARCHIVE_EVIDENCE_COPY_MISMATCH")
        evidence_identities[relative.as_posix()] = digest(path)
        git("add", "--", copied.relative_to(REPOSITORY).as_posix())
    report["evidence_files"] = evidence_identities
    after_index = index_entries()
    report["existing_index_entries_unchanged"] = all(
        after_index.get(path) == identity for path, identity in before_index.items())
    if not report["existing_index_entries_unchanged"]:
        raise SystemExit("PREEXISTING_INDEX_CHANGED")
    report["staged_count_after"] = sum(bool(path) for path in
                                        git("diff", "--cached", "--name-only", "-z").split(b"\0"))
    (EVIDENCE / "local-archive-status.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "packages"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
