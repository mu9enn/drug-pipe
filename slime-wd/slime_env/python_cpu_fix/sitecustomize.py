import os as _os

_orig_cpu_count = _os.cpu_count

def _fixed_cpu_count():
    v = _os.environ.get("PYTHON_CPU_COUNT")
    if v:
        try:
            n = int(v)
            if n >= 1:
                return n
        except Exception:
            pass

    try:
        return max(1, len(_os.sched_getaffinity(0)))
    except Exception:
        return _orig_cpu_count() or 1

_os.cpu_count = _fixed_cpu_count

try:
    import multiprocessing as _mp
    _mp.cpu_count = _fixed_cpu_count
except Exception:
    pass
