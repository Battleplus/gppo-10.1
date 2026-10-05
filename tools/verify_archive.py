"""Read-only, standard-library verification of this documentation archive."""
from pathlib import Path
import hashlib
import json
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    errors = []
    inventory = json.loads((ROOT/'provenance/source-inventory.json').read_text(encoding='utf-8'))
    history = json.loads((ROOT/'provenance/history-source-inventory.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT/'provenance/archive-hashes.json').read_text(encoding='utf-8'))
    for row in inventory['files'] + history['files'] + manifest['files']:
        rel = row.get('archived_path', row.get('path'))
        path = ROOT/rel
        if not path.is_file():
            errors.append('missing: '+rel)
        elif path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            errors.append('identity mismatch: '+rel)
    expected = {row['path'] for row in manifest['files']} | {'provenance/archive-hashes.json'}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
              and not any(x in {'.git', '.publish', '__pycache__'} for x in p.relative_to(ROOT).parts)}
    if expected != actual:
        errors.append('manifest file set mismatch: '+repr(sorted(expected ^ actual)))
    docs = [ROOT/'README.md', ROOT/'evidence/README.md', ROOT/'history-archive/README.md', *sorted((ROOT/'docs').glob('*.md'))]
    links = 0
    for doc in docs:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            target = unquote(target.split('#')[0])
            if not (doc.parent/target).exists():
                errors.append('broken link in '+doc.name+': '+target)
            links += 1
    print(json.dumps({'passed': not errors, 'archive_files': len(actual),
                      'verbatim_sources': len(inventory['files']) + len(history['files']), 'checked_links': links,
                      'errors': errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0

if __name__ == '__main__':
    sys.exit(main())
