"""Compatibility engineering entry; v5 uses the unified prefix acceptance suite."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('public_prefix_acceptance.py')), run_name='__main__')
