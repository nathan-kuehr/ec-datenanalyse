import os

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Memory
from tqdm import tqdm
from pyimpspec import set_default_num_procs

# Cache
Cache = Memory("./lab-analytics-cache", verbose=0)

# Flag for pyimpspec process use setup
__Num_Procs_Is_Set_Up = False


def multiprocess(
    method: Callable, input: list, tqdm_note: str = "", args: dict | list[dict] = {}
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

    with ProcessPoolExecutor(max_workers=(3 * (os.cpu_count() or 2) // 4)) as executor:
        # Get futures of the processed data
        futures = {
            executor.submit(method, sample, **arg): i
            for i, (sample, arg) in enumerate(zip(input, listed_args))
        }

        # Update progress bar as futures are filled
        for future in tqdm(
            as_completed(futures),
            total=len(input),
            desc=tqdm_note,
        ):
            results[futures[future]] = future.result()

    return results
