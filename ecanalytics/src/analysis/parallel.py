import os
import importlib

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Memory
from joblib.memory import MemorizedFunc
from tqdm import tqdm
from pyimpspec import set_default_num_procs

# Cache
Cache = Memory("./lab-analytics-cache", verbose=0)

# Flag for pyimpspec process use setup
__Num_Procs_Is_Set_Up = False

def _proxy_caller(method_ids: tuple, *args, **kwargs):
    module, fname = method_ids
    func = getattr(importlib.import_module(module), fname)
    return func(*args, **kwargs)

def multiprocess(
    method: MemorizedFunc, input: list, tqdm_note: str = "", args: dict | list[dict] = {}
) -> list:
    global __Num_Procs_Is_Set_Up

    # Set pyimpspec processes to 1, otherwise we get nested parallelization
    if not __Num_Procs_Is_Set_Up:
        set_default_num_procs(1)
        __Num_Procs_Is_Set_Up = True

    # Prepare list which stores the results in correct order
    results = [None] * len(input)

    # Check if we can use the same args for all samples
    if isinstance(args, dict):
        listed_args = [args] * len(input)
    else:
        listed_args = args

    # Prepare list for multiprocessing
    jobs_to_compute = []

    # Check if cached data exists
    for i, (sample, arg) in enumerate(zip(input, listed_args)):
        cached = method.check_call_in_cache(sample, **arg)

        if cached: 
            # Exists in cache -> just load
            results[i] = method(sample, **arg)
        else:
            # Append to job compute lsit
            jobs_to_compute.append((i, sample, arg))

    # Need to transform method 
    method_ids = method.func.__module__, method.func.__name__

    if jobs_to_compute:
        nprocs = (3 * (os.cpu_count() or 2) // 4)

        with ProcessPoolExecutor(max_workers=nprocs) as ex:
            # Get futures of the processed data
            futures = {ex.submit(_proxy_caller, method_ids, sample, **arg): i for i, sample, arg in jobs_to_compute}

            # Update progress bar as futures are filled
            for future in tqdm(
                as_completed(futures),
                total=len(jobs_to_compute),
                desc=tqdm_note,
            ):
                results[futures[future]] = future.result()

    return results
