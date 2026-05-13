from typing import Iterable

def call_argument_parser(
    args: tuple,
    kwargs: dict,
    default: dict,
    sample_names: Iterable[str],
    name: str = "",
) -> list:
    """Parses positional/keyword args for analysis-step calls.

    Either:
    - One positional dict mapping sample names to overrides, OR
    - Plain kwargs applied to all samples.

    Returns a list of merged-arg dicts, one per sample.
    """
    nargs, nkwargs = len(args), len(kwargs)

    if nargs > 0 and nkwargs == 0:
        if nargs > 1:
            raise ValueError(
                f"Only one positional argument allowed, but {nargs} were given."
            )
        if not isinstance(args[0], dict):
            raise ValueError(
                f"Please provide a dictionary mapping the sample names to the parameters to apply for the {name} calculation."
            )
        return [default | args[0].get(name, {}) for name in sample_names]

    if nargs == 0:
        return [(default | kwargs) for _ in sample_names]

    raise ValueError(
        "Please provide either only positional or only keyword arguments, not both."
    )
