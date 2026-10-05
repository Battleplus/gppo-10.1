import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runtime_preflight import RuntimeContractError, sha256_file, verify, verify_process_isolation, verify_record_payload


class PreflightTests(unittest.TestCase):
    @staticmethod
    def policy(prefix):
        return {"prefix": str(Path(prefix) / ".w1-no-bytecode-cache"), "must_remain_absent": True, "writes_disabled": True}

    def test_missing_process_flags_and_project_path_rejected(self):
        with patch("runtime_preflight.os.environ", {"PYTHONNOUSERSITE": "1"}), \
             patch.object(sys, "flags", SimpleNamespace(isolated=False, dont_write_bytecode=True, no_user_site=True)):
            with self.assertRaisesRegex(RuntimeContractError, "FLAGS_MISSING"):
                verify_process_isolation()
        with patch("runtime_preflight.os.environ", {"PYTHONNOUSERSITE": "1"}), \
             patch.object(sys, "flags", SimpleNamespace(isolated=True, dont_write_bytecode=True, no_user_site=True)), \
             patch.object(sys, "pycache_prefix", str(Path(sys.prefix) / ".w1-no-bytecode-cache")), \
             patch.object(sys, "path", ["/unexpected/project"]):
            with self.assertRaisesRegex(RuntimeContractError, "UNEXPECTED_PYTHON_SEARCH_PATH"):
                verify_process_isolation()

    def test_record_payload_detects_python_file_tamper(self):
        import base64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module = root / "module.py"
            module.write_bytes(b"module bytes")
            recorded = base64.urlsafe_b64encode(hashlib.sha256(module.read_bytes()).digest()).rstrip(b"=").decode()
            (root / "RECORD").write_text(f"module.py,sha256={recorded},12\n")
            distribution = SimpleNamespace(_path=root, locate_file=lambda relative: root / relative)
            verify_record_payload(distribution, root)
            module.write_bytes(b"broken bytes")
            with self.assertRaisesRegex(RuntimeContractError, "PAYLOAD_DIGEST_MISMATCH"):
                verify_record_payload(distribution, root)

    def test_existing_bytecode_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / ".w1-no-bytecode-cache"
            cache.mkdir()
            with patch("runtime_preflight.os.environ", {"PYTHONNOUSERSITE": "1"}), \
                 patch.object(sys, "prefix", str(root)), patch.object(sys, "pycache_prefix", str(cache)), \
                 patch.object(sys, "flags", SimpleNamespace(isolated=True, dont_write_bytecode=True, no_user_site=True)):
                with self.assertRaisesRegex(RuntimeContractError, "BYTECODE_CACHE_READ_ISOLATION_REQUIRED"):
                    verify_process_isolation()

    def test_loadable_bytecode_outside_cache_requires_a_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module = root / "module.pyc"
            module.write_bytes(b"loadable test bytes")
            (root / "RECORD").write_text("module.pyc,,\n")
            distribution = SimpleNamespace(_path=root, locate_file=lambda relative: root / relative)
            with self.assertRaisesRegex(RuntimeContractError, "UNHASHED_LOADABLE_BYTECODE_NOT_ALLOWED"):
                verify_record_payload(distribution, root)

    def test_runtime_identity_passes_without_tensor_or_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "python"
            library = root / "library.so"
            binary.write_bytes(b"test interpreter")
            library.write_bytes(b"test library")
            spec = {"schema": "w1-remote-runtime-identity/1.0.0", "python_executable": str(binary), "python_prefix": str(root), "bytecode_cache_policy": self.policy(root),
                    "python_binary_sha256": sha256_file(binary),
                    "installed_distribution_identities": {}, "critical_files": {str(library): sha256_file(library)},
                    "gpu_identities": [["GPU-one", "test GPU", "535"], ["GPU-two", "test GPU", "535"]]}
            gpu = subprocess.CompletedProcess([], 0, "GPU-one, test GPU, 535, 1000\nGPU-two, test GPU, 535, 1000\n", "")
            with patch.object(sys, "executable", str(binary)), patch.object(sys, "prefix", str(root)), \
                 patch("runtime_preflight.os.environ", {}), \
                 patch("runtime_preflight.importlib.metadata.distributions", return_value=[]), \
                 patch("runtime_preflight.subprocess.run", return_value=gpu) as command:
                result = verify(spec)
                self.assertEqual(result["status"], "pass")
                self.assertFalse(result["worker_started"])
                self.assertEqual(result["tensor_calls"], 0)
                self.assertEqual(command.call_count, 1)
                library.write_bytes(b"tampered library")
                with self.assertRaisesRegex(RuntimeContractError, "RUNTIME_FILE_DIGEST_MISMATCH"):
                    verify(spec)
                self.assertEqual(command.call_count, 1)

    def test_wrong_interpreter_stops_before_hardware_probe(self):
        with patch("runtime_preflight.subprocess.run") as command:
            with self.assertRaisesRegex(RuntimeContractError, "EXPLICIT_PYTHON_IDENTITY_MISMATCH"):
                verify({"schema": "w1-remote-runtime-identity/1.0.0", "python_executable": "/not/the/interpreter", "python_prefix": "/not/the/prefix", "bytecode_cache_policy": self.policy("/not/the/prefix")})
            command.assert_not_called()

    def test_dirty_cuda_search_path_stops_before_hardware_probe(self):
        with patch("runtime_preflight.os.environ", {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64"}), \
             patch("runtime_preflight.subprocess.run") as command:
            spec = {"schema": "w1-remote-runtime-identity/1.0.0", "python_executable": sys.executable, "python_prefix": sys.prefix, "bytecode_cache_policy": self.policy(sys.prefix),
                    "python_binary_sha256": sha256_file(Path(sys.executable).resolve())}
            with self.assertRaisesRegex(RuntimeContractError, "IMPLICIT_PATH_NOT_ALLOWED"):
                verify(spec)
            command.assert_not_called()

    def test_extra_distribution_stops_before_hardware_probe(self):
        with patch("runtime_preflight.os.environ", {}), \
             patch("runtime_preflight.importlib.metadata.distributions", return_value=[
                 SimpleNamespace(metadata={"Name": "unexpected"}, version="1")]), \
             patch("runtime_preflight.subprocess.run") as command:
            spec = {"schema": "w1-remote-runtime-identity/1.0.0", "python_executable": sys.executable, "python_prefix": sys.prefix, "bytecode_cache_policy": self.policy(sys.prefix),
                    "python_binary_sha256": sha256_file(Path(sys.executable).resolve()),
                    "installed_distribution_identities": {}}
            with self.assertRaisesRegex(RuntimeContractError, "DEPENDENCY_SET_MISMATCH"):
                verify(spec)
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
