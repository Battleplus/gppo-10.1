from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = {}
    for path in sorted(ROOT.iterdir()):
        if not path.is_file() or path.name == "hashes.json":
            continue
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    payload = {"schema": "w1-eawm-jepa-hashes/1.0.0", "package": ROOT.name, "files": files}
    (ROOT / "hashes.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "hashes_json": sha(ROOT / "hashes.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
