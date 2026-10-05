# Preserved offline diagnostic error

First command: `python -B capture_native_final.py`, exit code 0.
The script read the terminal artifacts and PID JSON, but its subprocess stderr
reader thread failed to decode WSL bytes using the Windows default GBK codec.
The original `native-final-status-evidence.json` is preserved and is incomplete
for stderr. Do not call that evidence complete, despite its process exit code.

```text
Exception in thread Thread-2 (_readerthread):
  C:\Python314\Lib\threading.py:1082, in _bootstrap_inner
  C:\Python314\Lib\threading.py:1024, in run
  C:\Python314\Lib\subprocess.py:1614, in _readerthread
    buffer.append(fh.read())
UnicodeDecodeError: 'gbk' codec can't decode byte 0xff in position 47: illegal multibyte sequence
```

The diagnostic was repaired to capture raw subprocess stdout/stderr bytes,
store their hexadecimal identities and explicitly decode stdout as UTF-8.
The second readonly invocation writes `native-final-status-evidence-v2.json`;
it does not rewrite the first evidence file or any source attempt artifact.
No research worker, environment, model, checkpoint or attempt was started by
either invocation. This was an audit-tool error, not an additional experiment.
