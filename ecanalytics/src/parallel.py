import os
import importlib

from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Memory
from joblib.memory import MemorizedFunc
from tqdm import tqdm
from pyimpspec import set_default_num_procs


CACHE = Memory("./lab-analytics-cache", verbose=0)

_PROCESS_POOL_FRACTION = 3 / 4
_DEFAULT_FALLBACK_CPU_COUNT = 2

_num_procs_is_set_up = False


def _proxy_caller(method_ids: tuple[str, str], *args, **kwargs):
    module, fname = method_ids
    func = getattr(importlib.import_module(module), fname)
    return func(*args, **kwargs)


def multiprocess(
    method: MemorizedFunc,
    inputs: list,
    tqdm_note: str = "",
    args: dict | list[dict] = {},
) -> list:
    global _num_procs_is_set_up

    # Set pyimpspec processes to 1, otherwise we get nested parallelization
    if not _num_procs_is_set_up:
        set_default_num_procs(1)
        _num_procs_is_set_up = True

    results: list = [None] * len(inputs)

    if isinstance(args, dict):
        listed_args = [args] * len(inputs)
    else:
        listed_args = args

    jobs_to_compute = []

    # Check if cached data exists
    for i, (sample, arg) in enumerate(zip(inputs, listed_args)):
        cached = method.check_call_in_cache(sample, **arg)

        if cached:
            results[i] = method(sample, **arg)
        else:
            jobs_to_compute.append((i, sample, arg))

    method_ids = method.func.__module__, method.func.__name__

    if jobs_to_compute:
        nprocs = int(_PROCESS_POOL_FRACTION * (os.cpu_count() or _DEFAULT_FALLBACK_CPU_COUNT))

        with ProcessPoolExecutor(max_workers=nprocs) as executor:
            futures = {
                executor.submit(_proxy_caller, method_ids, sample, **arg): i
                for i, sample, arg in jobs_to_compute
            }

            for future in tqdm(
                as_completed(futures),
                total=len(jobs_to_compute),
                desc=tqdm_note,
            ):
                results[futures[future]] = future.result()

    return results
