"""Actual no-model acceptance at the production worker boundary."""
import io
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parent

def run(evidence):
    import torch
    from isolated_fixture_preflight import verify
    from acceptance_entry import _write_json
    from manifest_contract import verify_package
    if evidence is None or torch.cuda.is_initialized():
        raise RuntimeError("SYSTEM_ACCEPTANCE_EXPECTS_NO_CUDA")
    evidence = Path(evidence).resolve()
    if evidence != ROOT.parent / "evidence" or evidence.exists():
        raise RuntimeError("SYSTEM_EVIDENCE_UNUSED_ISOLATED_PATH_REQUIRED")
    evidence.mkdir(mode=0o700)
    identity = verify_package(ROOT)
    started, cpu = time.monotonic(), time.process_time()
    old_tempdir, old_evidence = tempfile.tempdir, os.environ.get("W1_TEST_EVIDENCE_DIR")
    work = evidence / "system-work"
    work.mkdir()
    tempfile.tempdir = str(work)
    os.environ["W1_TEST_EVIDENCE_DIR"] = str(evidence)
    os.environ.pop("W1_PHASE_ACCOUNTING_SOCKET", None)
    original_module_init, original_cuda_init = torch.nn.Module.__init__, torch.cuda.init
    def reject_model(*args, **kwargs):
        raise RuntimeError("SYSTEM_ACCEPTANCE_MODEL_CONSTRUCTION_FORBIDDEN")
    def reject_cuda(*args, **kwargs):
        raise RuntimeError("SYSTEM_ACCEPTANCE_CUDA_INIT_FORBIDDEN")
    torch.nn.Module.__init__, torch.cuda.init = reject_model, reject_cuda
    result = {"classification": "NO_MODEL_SYSTEM_ACCEPTANCE", "formal_attempt_created": False,
        "engineering_job_consumed": False, "training_completion": "not_run",
        "package_identity": identity["hashes_sha256"], "python": sys.executable}
    try:
        result["imports"] = verify()
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            "test_phase_handshake", "test_supervised_phase_integration", "test_cpu_scope_contract",
            "test_server_acceptance_entry",
            "test_dependency_closure.DependencyClosureTests.test_missing_staged_transitive_module_fails_before_environment_construction",
            "test_dependency_closure.DependencyClosureTests.test_portable_sources_do_not_borrow_external_workspace",
            "test_dependency_closure.DependencyClosureTests.test_import_error_is_reported_before_worker_can_construct_environment"])
        stream = io.StringIO()
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        (evidence / "system-tests.log").write_text(stream.getvalue(), encoding="utf-8")
        result.update(tests_run=tests.testsRun, failures=len(tests.failures), errors=len(tests.errors),
            skipped=len(tests.skipped), status="complete" if tests.wasSuccessful() and not tests.skipped else "technical_stop")
        result["first_errors"] = [body for _, body in tests.errors + tests.failures]
    except BaseException:
        result["status"], result["exception_chain"] = "technical_stop", traceback.format_exc()
    finally:
        torch.nn.Module.__init__, torch.cuda.init = original_module_init, original_cuda_init
        tempfile.tempdir = old_tempdir
        if old_evidence is None:
            os.environ.pop("W1_TEST_EVIDENCE_DIR", None)
        else:
            os.environ["W1_TEST_EVIDENCE_DIR"] = old_evidence
    result.update(cuda_initialized=torch.cuda.is_initialized(), model_initializations=0,
        model_forwards=0, model_backward_calls=0, optimizer_updates=0, checkpoint_calls=0,
        real_environment_calls=0, wall_seconds=time.monotonic()-started, self_cpu_seconds=time.process_time()-cpu)
    if result["cuda_initialized"]:
        result["status"] = "technical_stop"
    _write_json(evidence / "acceptance-result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "complete" else 1
