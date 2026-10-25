"""Paths and JSON input/output shared by the experiment scripts.

Importing this module puts the project root first on sys.path, so that an
experiment run from a copy of the project uses that copy's library.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def processor_name() -> str:
    """Model name of the processor, for artifacts that record timings."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    import platform
    return platform.processor() or platform.machine()


def path(name: str) -> str:
    """Absolute path of results/<name>."""
    return os.path.join(RESULTS, name)


def exists(name: str) -> bool:
    return os.path.exists(path(name))


def load(name: str):
    with open(path(name), encoding="utf-8") as fh:
        return json.load(fh)


def save(name: str, payload, indent: int = 1, quiet: bool = False) -> str:
    """Write results/<name> as UTF-8 JSON with LF line endings.

    The file is written to a temporary name and then renamed, so an
    interrupted write never leaves a truncated artifact.
    """
    p = path(name)
    with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=indent)
        fh.write("\n")
    os.replace(p + ".tmp", p)
    if not quiet:
        print(f"wrote results/{name}")
    return p
