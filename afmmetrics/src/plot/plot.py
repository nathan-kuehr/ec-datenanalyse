from matplotlib import pyplot as plt

from ..data.afm import AFMImage
from ..config import FIGURE_SETTINGS


def history(afm: AFMImage) -> None:
    hist = list(afm._history.values())
    nhist = len(hist)

    ncols = min(4, nhist)
    nrows = 1 + (nhist - 1) // 4

    if len(hist) > nrows * ncols:
        raise ValueError(
            "Shape needs to be sufficiently large for the AFM image's modification history."
        )

    with plt.rc_context(FIGURE_SETTINGS):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False
        )

        for i, ax in enumerate(axes.flatten()):
            # Check if we have history items left for this subplot
            if i < len(hist):
                img, desc = hist[i]
                if (dims := len(img.shape)) == 3:
                    ax.imshow(img)
                elif dims == 2:
                    ax.imshow(img, cmap="gray")
                else:
                    raise RuntimeError(
                        f"Unexpected image dimensions: {dims}. Expected 2 or 3."
                    )

                ax.set_title(desc)

            # Hide axis for all
            ax.axis("off")

        fig.tight_layout()

    plt.show()
