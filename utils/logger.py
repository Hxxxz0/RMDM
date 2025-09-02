import os

_ACC = {}
_DIR = None


def configure(dir: str):
    global _DIR
    _DIR = dir
    os.makedirs(_DIR, exist_ok=True)


def logkv_mean(key, val):
    _ACC[key] = float(val)


def get_dir():
    return _DIR or '.'


def log(msg):
    print(msg)


def dumpkvs():
    if not _ACC:
        return
    print({k: round(v, 6) for k, v in _ACC.items()})
    _ACC.clear()


