"""Finalize small offline artifacts and preserve every pre-existing index entry."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from analyze_v6 import ATTEMPT, PACKAGE, read_json, require, sha, snapshot, write_json


FILES = (
    "analysis-plan.md", "analysis.json", "analyze_v6.py", "attribution-erratum.json",
    "candidate-table.csv", "capture_native_final.py", "decision.json",
    "luna-label-review.md", "luna-resource-review.md", "native-final-status-evidence-v2.json",
    "native-final-status-evidence.json", "native-read-decoding-error.md",
    "parent-split-proposal.json", "prepare_collection_proposal.py", "previous-claim-erratum.json",
    "report.md", "RESOURCE_REQUEST-proposal.json", "source-identities-after.json",
    "source-identities-before.json", "test_offline_acceptance.py", "test-evidence.json",
    "finalize_archive.py", "completion-audit.json",
)


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True, timeout=30).stdout


def index(repo):
    rows = {}
    for entry in git(repo, "ls-files", "--stage", "-z").split(b"\0"):
        if entry:
            identity, path = entry.split(b"\t", 1)
            rows[path] = identity
    return rows


def main():
    source = Path(__file__).resolve().parent
    workspace = source.parents[1]
    package = workspace / "research-plans" / PACKAGE
    attempt = workspace / "runs" / ATTEMPT
    current = {"package": snapshot(package), "attempt": snapshot(attempt)}
    require(current == read_json(source / "source-identities-after.json"), "old_evidence_changed_since_analysis")
    request = read_json(source / "RESOURCE_REQUEST-proposal.json")
    require(request["status"] == "NOT_APPROVED" and request["runner_ready"] is False, "proposal_not_fail_closed")
    require(not (workspace / "runs" / request["proposed_attempt"]).exists(), "proposed_attempt_created")
    for filename in FILES:
        require((source / filename).is_file() and (source / filename).stat().st_size < 200000, "missing_or_large_archive_file:" + filename)
    repo = workspace / "research-progress-archive"
    destination = repo / "docs/archive/w1-label-v6-offline-acceptance-v1"
    require(not destination.exists(), "archive_path_already_exists_do_not_overwrite")
    before = index(repo)
    destination.mkdir(parents=True)
    (destination / ".gitattributes").write_bytes(b"* -text\n")
    for filename in FILES:
        shutil.copyfile(source / filename, destination / filename)
    relative = destination.relative_to(repo).as_posix()
    git(repo, "add", "--", relative)
    after = index(repo)
    require(all(after.get(path) == identity for path, identity in before.items()), "preexisting_index_entries_changed")
    for filename in FILES:
        original = (source / filename).read_bytes()
        staged = git(repo, "show", ":" + relative + "/" + filename)
        require(original == (destination / filename).read_bytes() == staged, "archive_byte_identity:" + filename)
    status = {"local_archive": str(destination), "local_small_files_copied_and_staged": True,
              "preexisting_index_entry_count": len(before), "preexisting_index_entries_unchanged": True,
              "commit_gpgsign": True, "gpg_format": "ssh", "signing_agent": "Disabled/Stopped",
              "new_commit_created": False, "push_attempted": False, "remote_archived": False,
              "remote_read_this_turn": {"command": "git ls-remote origin refs/heads/main", "exit_code": 1,
                                       "stderr": "error: RPC failed; curl 56 Recv failure: Connection was reset\nfatal: expected flush after ref listing\n"},
              "last_previously_verified_remote_main": "6aa16410b0e69aaceb7dee0b4d260eaf76d03e02",
              "last_sha_is_not_a_current_verification_or_this_delivery_commit": True,
              "secrets_raw_data_sqlite_checkpoints_uploaded": False,
              "next_external_action": "Restore the authorized SSH signing agent and existing signing key, then recheck remote changes and safely create a signed archive commit. Do not disable signing or force-push."}
    write_json(source / "archive-status.json", status)
    hashes = {name: sha((source / name).read_bytes()) for name in FILES + ("archive-status.json",)}
    write_json(source / "delivery-hashes.json", {"schema": "w1-offline-label-acceptance-delivery/1.0.0",
                                               "purpose": "offline analysis evidence, not an experimental execution manifest",
                                               "files": hashes, "source_data_sha256": read_json(source / "analysis.json")["source_data_sha256"]})
    for filename in ("archive-status.json", "delivery-hashes.json"):
        shutil.copyfile(source / filename, destination / filename)
    git(repo, "add", "--", relative)
    final_index = index(repo)
    require(all(final_index.get(path) == identity for path, identity in before.items()), "preexisting_index_changed_after_finalization")
    for filename in FILES + ("archive-status.json", "delivery-hashes.json"):
        require((source / filename).read_bytes() == (destination / filename).read_bytes() == git(repo, "show", ":" + relative + "/" + filename), "final_archive_identity:" + filename)
    require(current == {"package": snapshot(package), "attempt": snapshot(attempt)}, "old_evidence_changed_during_archiving")
    print(json.dumps({"status": "pass", "old_sources_unchanged": True, "archive_bytes_match_disk_and_index": True,
                      "preserved_existing_index_entries": len(before), "final_index_entries": len(final_index),
                      "delivery_hashes_sha256": sha((source / "delivery-hashes.json").read_bytes()),
                      "remote_archived": False, "formal_attempt_created": False}))


if __name__ == "__main__":
    main()
